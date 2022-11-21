//!
//! WAL redo. This service runs PostgreSQL in a special wal_redo mode
//! to apply given WAL records over an old page image and return new
//! page image.
//!
//! We rely on Postgres to perform WAL redo for us. We launch a
//! postgres process in special "wal redo" mode that's similar to
//! single-user mode. We then pass the previous page image, if any,
//! and all the WAL records we want to apply, to the postgres
//! process. Then we get the page image back. Communication with the
//! postgres process happens via stdin/stdout
//!
//! See pgxn/neon_walredo/walredoproc.c for the other side of
//! this communication.
//!
//! The Postgres process is assumed to be secure against malicious WAL
//! records. It achieves it by dropping privileges before replaying
//! any WAL records, so that even if an attacker hijacks the Postgres
//! process, he cannot escape out of it.
//!
use anyhow::Context;
use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::Serialize;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tracing::*;
use utils::crashsafe::path_with_suffix_extension;
use utils::{bin_ser::BeSer, id::TenantId, lsn::Lsn};

use crate::metrics::{
    WAL_REDO_BYTES_HISTOGRAM, WAL_REDO_RECORDS_HISTOGRAM, WAL_REDO_RECORD_COUNTER, WAL_REDO_TIME,
};
use crate::pgdatadir_mapping::{key_to_rel_block, key_to_slru_block};
use crate::repository::Key;
use crate::task_mgr::BACKGROUND_RUNTIME;
use crate::walrecord::NeonWalRecord;
use crate::{config::PageServerConf, TEMP_FILE_SUFFIX};
use pageserver_api::reltag::{RelTag, SlruKind};
use postgres_ffi::pg_constants;
use postgres_ffi::relfile_utils::VISIBILITYMAP_FORKNUM;
use postgres_ffi::v14::nonrelfile_utils::{
    mx_offset_to_flags_bitshift, mx_offset_to_flags_offset, mx_offset_to_member_offset,
    transaction_id_set_status,
};
use postgres_ffi::BLCKSZ;

///
/// `RelTag` + block number (`blknum`) gives us a unique id of the page in the cluster.
///
/// In Postgres `BufferTag` structure is used for exactly the same purpose.
/// [See more related comments here](https://github.com/postgres/postgres/blob/99c5852e20a0987eca1c38ba0c09329d4076b6a0/src/include/storage/buf_internals.h#L91).
///
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize)]
pub struct BufferTag {
    pub rel: RelTag,
    pub blknum: u32,
}

impl BufferTag {
    /// Serialized length
    pub const LEN: u32 = RelTag::LEN + 4;
}

///
/// WAL Redo Manager is responsible for replaying WAL records.
///
/// Callers use the WAL redo manager through this abstract interface,
/// which makes it easy to mock it in tests.
pub trait WalRedoManager: Send + Sync {
    /// Apply some WAL records.
    ///
    /// The caller passes an old page image, and WAL records that should be
    /// applied over it. The return value is a new page image, after applying
    /// the reords.
    fn request_redo(
        &self,
        key: Key,
        lsn: Lsn,
        base_img: Option<Bytes>,
        records: Vec<(Lsn, NeonWalRecord)>,
        pg_version: u32,
    ) -> Result<Bytes, WalRedoError>;
}

///
/// This is the real implementation that uses a Postgres process to
/// perform WAL replay. Only one thread can use the process at a time,
/// that is controlled by the Mutex. In the future, we might want to
/// launch a pool of processes to allow concurrent replay of multiple
/// records.
///
pub struct PostgresRedoManager {
    conf: &'static PageServerConf,
    handle: Handle,
}

/// Can this request be served by neon redo functions
/// or we need to pass it to wal-redo postgres process?
fn can_apply_in_neon(rec: &NeonWalRecord) -> bool {
    // Currently, we don't have bespoken Rust code to replay any
    // Postgres WAL records. But everything else is handled in neon.
    #[allow(clippy::match_like_matches_macro)]
    match rec {
        NeonWalRecord::Postgres {
            will_init: _,
            rec: _,
        } => false,
        _ => true,
    }
}

/// An error happened in WAL redo
#[derive(Debug, thiserror::Error)]
pub enum WalRedoError {
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("cannot perform WAL redo now")]
    InvalidState,
    #[error("cannot perform WAL redo for this request")]
    InvalidRequest,
    #[error("cannot perform WAL redo for this record")]
    InvalidRecord,
}

///
/// Public interface of WAL redo manager
///
impl WalRedoManager for PostgresRedoManager {
    ///
    /// Request the WAL redo manager to apply some WAL records
    ///
    /// The WAL redo is handled by a separate thread, so this just sends a request
    /// to the thread and waits for response.
    ///
    fn request_redo(
        &self,
        key: Key,
        lsn: Lsn,
        base_img: Option<Bytes>,
        records: Vec<(Lsn, NeonWalRecord)>,
        pg_version: u32,
    ) -> Result<Bytes, WalRedoError> {
        if records.is_empty() {
            error!("invalid WAL redo request with no records");
            return Err(WalRedoError::InvalidRequest);
        }

        // convert it to an arc to avoid cloning it on batches
        let records: Arc<[(Lsn, NeonWalRecord)]> = records.into();

        let mut img: Option<Bytes> = base_img;
        let mut batch_neon = can_apply_in_neon(&records[0].1);
        let mut batch_start = 0;
        for i in 1..records.len() {
            let rec_neon = can_apply_in_neon(&records[i].1);

            if rec_neon != batch_neon {
                let result = if batch_neon {
                    self.apply_batch_neon(key, lsn, img, &records[batch_start..i])
                } else {
                    self.apply_batch_postgres(
                        key,
                        lsn,
                        img,
                        &records,
                        (batch_start..i).into(),
                        self.conf.wal_redo_timeout,
                        pg_version,
                    )
                };
                img = Some(result?);

                batch_neon = rec_neon;
                batch_start = i;
            }
        }
        // last batch
        if batch_neon {
            self.apply_batch_neon(key, lsn, img, &records[batch_start..])
        } else {
            self.apply_batch_postgres(
                key,
                lsn,
                img,
                &records,
                (batch_start..).into(),
                self.conf.wal_redo_timeout,
                pg_version,
            )
        }
    }
}

impl PostgresRedoManager {
    ///
    /// Create a new PostgresRedoManager.
    ///
    pub fn new(conf: &'static PageServerConf, tenant_id: TenantId) -> PostgresRedoManager {
        // The actual process is launched lazily, on first request.

        let (handle, fut) = tokio_postgres_redo(conf, tenant_id, 14);
        BACKGROUND_RUNTIME.spawn(fut);

        PostgresRedoManager { conf, handle }
    }

    ///
    /// Process one request for WAL redo using wal-redo postgres
    ///
    #[allow(clippy::too_many_arguments)]
    fn apply_batch_postgres(
        &self,
        key: Key,
        lsn: Lsn,
        base_img: Option<Bytes>,
        records: &Arc<[(Lsn, NeonWalRecord)]>,
        records_range: SliceRange,
        wal_redo_timeout: Duration,
        _pg_version: u32,
    ) -> Result<Bytes, WalRedoError> {
        let (rel, blknum) = key_to_rel_block(key).or(Err(WalRedoError::InvalidRecord))?;

        let start_time = Instant::now();

        // Relational WAL records are applied using wal-redo-postgres
        let buf_tag = BufferTag { rel, blknum };

        let record_count = records_range.sub_slice(records).len() as u64;

        let result = BACKGROUND_RUNTIME
            .block_on(self.handle.request_redo(Request {
                target: buf_tag,
                base_img,
                records: records.clone(),
                records_range,
                timeout: wal_redo_timeout,
            }))
            .map_err(|e| WalRedoError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)));

        let duration = start_time.elapsed();

        let len = records.len();
        let nbytes = records.iter().fold(0, |acumulator, record| {
            acumulator
                + match &record.1 {
                    NeonWalRecord::Postgres { rec, .. } => rec.len(),
                    _ => unreachable!("Only PostgreSQL records are accepted in this batch"),
                }
        });

        WAL_REDO_TIME.observe(duration.as_secs_f64());
        WAL_REDO_RECORDS_HISTOGRAM.observe(len as f64);
        WAL_REDO_BYTES_HISTOGRAM.observe(nbytes as f64);
        WAL_REDO_RECORD_COUNTER.inc_by(record_count);

        debug!(
            "postgres applied {} WAL records ({} bytes) in {} us to reconstruct page image at LSN {}",
            len,
            nbytes,
            duration.as_micros(),
            lsn
        );

        result
    }

    ///
    /// Process a batch of WAL records using bespoken Neon code.
    ///
    fn apply_batch_neon(
        &self,
        key: Key,
        lsn: Lsn,
        base_img: Option<Bytes>,
        records: &[(Lsn, NeonWalRecord)],
    ) -> Result<Bytes, WalRedoError> {
        let start_time = Instant::now();

        let mut page = BytesMut::new();
        if let Some(fpi) = base_img {
            // If full-page image is provided, then use it...
            page.extend_from_slice(&fpi[..]);
        } else {
            // All the current WAL record types that we can handle require a base image.
            error!("invalid neon WAL redo request with no base image");
            return Err(WalRedoError::InvalidRequest);
        }

        // Apply all the WAL records in the batch
        for (record_lsn, record) in records.iter() {
            self.apply_record_neon(key, &mut page, *record_lsn, record)?;
        }
        // Success!
        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);
        WAL_REDO_TIME.observe(duration.as_secs_f64());

        debug!(
            "neon applied {} WAL records in {} ms to reconstruct page image at LSN {}",
            records.len(),
            duration.as_micros(),
            lsn
        );

        Ok(page.freeze())
    }

    fn apply_record_neon(
        &self,
        key: Key,
        page: &mut BytesMut,
        _record_lsn: Lsn,
        record: &NeonWalRecord,
    ) -> Result<(), WalRedoError> {
        match record {
            NeonWalRecord::Postgres {
                will_init: _,
                rec: _,
            } => {
                error!("tried to pass postgres wal record to neon WAL redo");
                return Err(WalRedoError::InvalidRequest);
            }
            NeonWalRecord::ClearVisibilityMapFlags {
                new_heap_blkno,
                old_heap_blkno,
                flags,
            } => {
                // sanity check that this is modifying the correct relation
                let (rel, blknum) = key_to_rel_block(key).or(Err(WalRedoError::InvalidRecord))?;
                assert!(
                    rel.forknum == VISIBILITYMAP_FORKNUM,
                    "ClearVisibilityMapFlags record on unexpected rel {}",
                    rel
                );
                if let Some(heap_blkno) = *new_heap_blkno {
                    // Calculate the VM block and offset that corresponds to the heap block.
                    let map_block = pg_constants::HEAPBLK_TO_MAPBLOCK(heap_blkno);
                    let map_byte = pg_constants::HEAPBLK_TO_MAPBYTE(heap_blkno);
                    let map_offset = pg_constants::HEAPBLK_TO_OFFSET(heap_blkno);

                    // Check that we're modifying the correct VM block.
                    assert!(map_block == blknum);

                    // equivalent to PageGetContents(page)
                    let map = &mut page[pg_constants::MAXALIGN_SIZE_OF_PAGE_HEADER_DATA..];

                    map[map_byte as usize] &= !(flags << map_offset);
                }

                // Repeat for 'old_heap_blkno', if any
                if let Some(heap_blkno) = *old_heap_blkno {
                    let map_block = pg_constants::HEAPBLK_TO_MAPBLOCK(heap_blkno);
                    let map_byte = pg_constants::HEAPBLK_TO_MAPBYTE(heap_blkno);
                    let map_offset = pg_constants::HEAPBLK_TO_OFFSET(heap_blkno);

                    assert!(map_block == blknum);

                    let map = &mut page[pg_constants::MAXALIGN_SIZE_OF_PAGE_HEADER_DATA..];

                    map[map_byte as usize] &= !(flags << map_offset);
                }
            }
            // Non-relational WAL records are handled here, with custom code that has the
            // same effects as the corresponding Postgres WAL redo function.
            NeonWalRecord::ClogSetCommitted { xids, timestamp } => {
                let (slru_kind, segno, blknum) =
                    key_to_slru_block(key).or(Err(WalRedoError::InvalidRecord))?;
                assert_eq!(
                    slru_kind,
                    SlruKind::Clog,
                    "ClogSetCommitted record with unexpected key {}",
                    key
                );
                for &xid in xids {
                    let pageno = xid as u32 / pg_constants::CLOG_XACTS_PER_PAGE;
                    let expected_segno = pageno / pg_constants::SLRU_PAGES_PER_SEGMENT;
                    let expected_blknum = pageno % pg_constants::SLRU_PAGES_PER_SEGMENT;

                    // Check that we're modifying the correct CLOG block.
                    assert!(
                        segno == expected_segno,
                        "ClogSetCommitted record for XID {} with unexpected key {}",
                        xid,
                        key
                    );
                    assert!(
                        blknum == expected_blknum,
                        "ClogSetCommitted record for XID {} with unexpected key {}",
                        xid,
                        key
                    );

                    transaction_id_set_status(
                        xid,
                        pg_constants::TRANSACTION_STATUS_COMMITTED,
                        page,
                    );
                }

                // Append the timestamp
                if page.len() == BLCKSZ as usize + 8 {
                    page.truncate(BLCKSZ as usize);
                }
                if page.len() == BLCKSZ as usize {
                    page.extend_from_slice(&timestamp.to_be_bytes());
                } else {
                    warn!(
                        "CLOG blk {} in seg {} has invalid size {}",
                        blknum,
                        segno,
                        page.len()
                    );
                }
            }
            NeonWalRecord::ClogSetAborted { xids } => {
                let (slru_kind, segno, blknum) =
                    key_to_slru_block(key).or(Err(WalRedoError::InvalidRecord))?;
                assert_eq!(
                    slru_kind,
                    SlruKind::Clog,
                    "ClogSetAborted record with unexpected key {}",
                    key
                );
                for &xid in xids {
                    let pageno = xid as u32 / pg_constants::CLOG_XACTS_PER_PAGE;
                    let expected_segno = pageno / pg_constants::SLRU_PAGES_PER_SEGMENT;
                    let expected_blknum = pageno % pg_constants::SLRU_PAGES_PER_SEGMENT;

                    // Check that we're modifying the correct CLOG block.
                    assert!(
                        segno == expected_segno,
                        "ClogSetAborted record for XID {} with unexpected key {}",
                        xid,
                        key
                    );
                    assert!(
                        blknum == expected_blknum,
                        "ClogSetAborted record for XID {} with unexpected key {}",
                        xid,
                        key
                    );

                    transaction_id_set_status(xid, pg_constants::TRANSACTION_STATUS_ABORTED, page);
                }
            }
            NeonWalRecord::MultixactOffsetCreate { mid, moff } => {
                let (slru_kind, segno, blknum) =
                    key_to_slru_block(key).or(Err(WalRedoError::InvalidRecord))?;
                assert_eq!(
                    slru_kind,
                    SlruKind::MultiXactOffsets,
                    "MultixactOffsetCreate record with unexpected key {}",
                    key
                );
                // Compute the block and offset to modify.
                // See RecordNewMultiXact in PostgreSQL sources.
                let pageno = mid / pg_constants::MULTIXACT_OFFSETS_PER_PAGE as u32;
                let entryno = mid % pg_constants::MULTIXACT_OFFSETS_PER_PAGE as u32;
                let offset = (entryno * 4) as usize;

                // Check that we're modifying the correct multixact-offsets block.
                let expected_segno = pageno / pg_constants::SLRU_PAGES_PER_SEGMENT;
                let expected_blknum = pageno % pg_constants::SLRU_PAGES_PER_SEGMENT;
                assert!(
                    segno == expected_segno,
                    "MultiXactOffsetsCreate record for multi-xid {} with unexpected key {}",
                    mid,
                    key
                );
                assert!(
                    blknum == expected_blknum,
                    "MultiXactOffsetsCreate record for multi-xid {} with unexpected key {}",
                    mid,
                    key
                );

                LittleEndian::write_u32(&mut page[offset..offset + 4], *moff);
            }
            NeonWalRecord::MultixactMembersCreate { moff, members } => {
                let (slru_kind, segno, blknum) =
                    key_to_slru_block(key).or(Err(WalRedoError::InvalidRecord))?;
                assert_eq!(
                    slru_kind,
                    SlruKind::MultiXactMembers,
                    "MultixactMembersCreate record with unexpected key {}",
                    key
                );
                for (i, member) in members.iter().enumerate() {
                    let offset = moff + i as u32;

                    // Compute the block and offset to modify.
                    // See RecordNewMultiXact in PostgreSQL sources.
                    let pageno = offset / pg_constants::MULTIXACT_MEMBERS_PER_PAGE as u32;
                    let memberoff = mx_offset_to_member_offset(offset);
                    let flagsoff = mx_offset_to_flags_offset(offset);
                    let bshift = mx_offset_to_flags_bitshift(offset);

                    // Check that we're modifying the correct multixact-members block.
                    let expected_segno = pageno / pg_constants::SLRU_PAGES_PER_SEGMENT;
                    let expected_blknum = pageno % pg_constants::SLRU_PAGES_PER_SEGMENT;
                    assert!(
                        segno == expected_segno,
                        "MultiXactMembersCreate record for offset {} with unexpected key {}",
                        moff,
                        key
                    );
                    assert!(
                        blknum == expected_blknum,
                        "MultiXactMembersCreate record for offset {} with unexpected key {}",
                        moff,
                        key
                    );

                    let mut flagsval = LittleEndian::read_u32(&page[flagsoff..flagsoff + 4]);
                    flagsval &= !(((1 << pg_constants::MXACT_MEMBER_BITS_PER_XACT) - 1) << bshift);
                    flagsval |= member.status << bshift;
                    LittleEndian::write_u32(&mut page[flagsoff..flagsoff + 4], flagsval);
                    LittleEndian::write_u32(&mut page[memberoff..memberoff + 4], member.xid);
                }
            }
        }

        Ok(())
    }
}

/// Serializes the wal redo request into `buffers` with the help of scratch buffer `scratch`.
///
/// The request is combination of `B + P + A* + G`.
///
/// Compared to [`build_vectored_messages`], this implementation builds at most 3 messages if the
/// base version of page is included (it's never copied to conserve the "scratch" space).
fn build_messages(
    target: BufferTag,
    base_img: Option<Bytes>,
    records: &[(Lsn, NeonWalRecord)],
    scratch: &mut BytesMut,
    buffers: &mut BufQueue,
) {
    target.ser_into(&mut scratch.writer()).unwrap();
    let tag = scratch.split().freeze();

    build_begin_message(&tag, scratch);

    if let Some(page) = base_img {
        assert_eq!(page.len(), 8192);
        build_push_page_header(&tag, scratch);

        let out = scratch.split().freeze();

        buffers.push(out);
        buffers.push(page);
    }

    for (end_lsn, record) in records {
        let (_will_init, rec) = match record {
            NeonWalRecord::Postgres { will_init, rec } => (will_init, rec),
            _ => unreachable!(),
        };

        build_apply_record_header(end_lsn, rec.len() as u32, scratch);
        buffers.push(scratch.split().freeze());
        buffers.push(rec.clone());
    }

    build_get_page_message(&tag, scratch);

    let out = scratch.split().freeze();
    buffers.push(out);
}

/// Compared to [`build_messages`] builds many small messages and aiming for vectored write
/// handling the gathering of the already allocated records.
fn build_vectored_messages(
    target: BufferTag,
    base_img: Option<Bytes>,
    records: &[(Lsn, NeonWalRecord)],
    scratch: &mut BytesMut,
    buffers: &mut BufQueue,
) {
    let tag = {
        target.ser_into(&mut scratch.writer()).unwrap();
        scratch.split().freeze()
    };

    build_begin_message(&tag, scratch);

    if let Some(page) = base_img {
        build_push_page_header(&tag, scratch);
        buffers.push(scratch.split().freeze());
        buffers.push(page);
    }

    for (end_lsn, record) in records {
        let rec = match record {
            NeonWalRecord::Postgres { rec, .. } => rec,
            _ => unreachable!(),
        };

        let record_len = rec.len() as u32;
        build_apply_record_header(end_lsn, record_len, scratch);
        buffers.push(scratch.split().freeze());
        buffers.push(rec.clone());
    }

    build_get_page_message(&tag, scratch);
    buffers.push(scratch.split().freeze());
}

fn build_begin_message(tag: &Bytes, scratch: &mut BytesMut) {
    scratch.put_u8(b'B');
    scratch.put_u32(4 + BufferTag::LEN);
    scratch.put(tag.clone());
}

fn build_push_page_header(tag: &Bytes, scratch: &mut BytesMut) {
    let page_len = 8192;
    scratch.put_u8(b'P');
    scratch.put_u32(4 + BufferTag::LEN + page_len);
    scratch.put(tag.clone());
}

fn build_apply_record_header(end_lsn: &Lsn, record_len: u32, scratch: &mut BytesMut) {
    scratch.put_u8(b'A');
    scratch.put_u32(4 + 8 + record_len);
    scratch.put_u64(end_lsn.0);
}

fn build_get_page_message(tag: &Bytes, scratch: &mut BytesMut) {
    scratch.put_u8(b'G');
    scratch.put_u32(4 + BufferTag::LEN);
    scratch.put(tag.clone());
}

/// Copypasted for tokio, need #[cfg(unix)]
trait TokioCloseFileDescriptors {
    fn close_fds(&mut self) -> &mut tokio::process::Command;
}

impl TokioCloseFileDescriptors for tokio::process::Command {
    fn close_fds(&mut self) -> &mut tokio::process::Command {
        unsafe {
            self.pre_exec(|| {
                // SAFETY: Code executed inside pre_exec should have async-signal-safety,
                // which means it should be safe to execute inside a signal handler.
                // The precise meaning depends on platform. See `man signal-safety`
                // for the linux definition.
                //
                // The set_fds_cloexec_threadsafe function is documented to be
                // async-signal-safe.
                //
                // Aside from this function, the rest of the code is re-entrant and
                // doesn't make any syscalls. We're just passing constants.
                //
                // NOTE: It's easy to indirectly cause a malloc or lock a mutex,
                // which is not async-signal-safe. Be careful.
                close_fds::set_fds_cloexec_threadsafe(3, &[]);
                Ok(())
            })
        }
    }
}

fn tokio_postgres_redo(
    conf: &'static PageServerConf,
    tenant_id: TenantId,
    pg_version: u32,
) -> (
    Handle,
    impl std::future::Future<Output = anyhow::Result<()>> + Send + 'static,
) {
    use tokio::io::AsyncWrite;

    // precise sizing for this would be pipe size divided by our average redo request
    let expected_inflight = 32;

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Payload>(expected_inflight);

    let ipc = async move {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};

        let mut buffers = BufQueue::default();
        let mut scratch = BytesMut::with_capacity(
            // without vectoring we aim at 3 messages: begin, page, records + get_page,
            // with vectoring this will be very much enough
            1024 * 8 * 3,
        );

        // loop to handle wal-redo process failing in between. additionally tenant_mgr expects that
        // walredo does not create the temporary directory until we get the first redo request, so
        // postpone creation until we get the first one.
        while let Some(first) = rx.recv().await {
            // make sure we dont have anything remaining from a past partial write
            buffers.clear();

            let mut child = launch_walredo(conf, tenant_id, pg_version).await?;
            let pid = child
                .id()
                .expect("pid is present before killing the process");

            info!("Launched wal-redo process for {tenant_id}: {pid}");

            // we send the external the request in different commands
            let mut stdin = child.stdin.take().expect("not taken yet");
            // stdout is used to communicate the resulting page
            let mut stdout = child.stdout.take().expect("not taken yet");
            // used to communicate hopefully utf-8 log messages
            let stderr = child.stderr.take().expect("not taken yet");
            let mut stderr = tokio::io::BufReader::new(stderr);

            let (result_txs, result_rx) = tokio::sync::mpsc::channel(expected_inflight);

            let have_vectored_stdin = stdin.is_write_vectored();

            let stdin_task = async {
                // cancellation safety is probably not an issue, vecdeque could be used to capture
                // oneshot senders

                let result_txs = result_txs;
                let mut buffered = Some(first);

                loop {
                    let (request, response) = {
                        // TODO: could we somehow manage to keep the request in case we need to
                        // restart the process? see https://github.com/neondatabase/neon/issues/1700
                        let next = buffered.take();
                        let next = if next.is_none() {
                            rx.recv().await
                        } else {
                            next
                        };

                        match next {
                            Some(t) => t,
                            None => break,
                        }
                    };

                    let records = request.records_range.sub_slice(&request.records);

                    if have_vectored_stdin {
                        build_vectored_messages(
                            request.target,
                            request.base_img,
                            records,
                            &mut scratch,
                            &mut buffers,
                        );
                    } else {
                        build_messages(
                            request.target,
                            request.base_img,
                            records,
                            &mut scratch,
                            &mut buffers,
                        );
                    }

                    let write_res = async {
                        while buffers.has_remaining() {
                            futures::future::poll_fn(|cx| {
                                tokio_util::io::poll_write_buf(
                                    std::pin::Pin::new(&mut stdin),
                                    cx,
                                    &mut buffers,
                                )
                            })
                            .await
                            .map_err(anyhow::Error::new)?;
                        }
                        // in general flush is not needed, does nothing on pipes
                        stdin.flush().await.map_err(anyhow::Error::new)
                    };

                    let slot = async { result_txs.reserve().await };

                    match tokio::join!(write_res, slot) {
                        (Ok(()), Ok(slot)) => {
                            // because we are pipelining, "start counting" the timeout only after we have
                            // written everything.
                            slot.send((response, tokio::time::Instant::now() + request.timeout));
                        }
                        (Ok(()), Err(_closed)) => {
                            drop(
                                response
                                    .send(Err(anyhow::anyhow!("Failed to receive the response"))),
                            );
                            return Err("stdout closed channel");
                        }
                        (Err(io), Ok(slot)) => {
                            drop(slot);
                            drop(
                                response
                                    .send(Err(io).context("Failed to write request to wal-redo")),
                            );
                            // we can still continue processing pipelined requests, if any. the
                            // stdout task will exit upon seeing we've dropped the result_txs.
                            return Ok(());
                        }
                        (Err(io), Err(_closed)) => {
                            drop(
                                response
                                    .send(Err(io).context("Failed to write request to wal-redo")),
                            );
                            return Err(
                                "io error while writing request while stdout closed channel",
                            );
                        }
                    };
                }

                // the Handle or the request queue sender have been dropped; return Ok(()) to keep
                // processing any of already pipelined requests
                Ok(())
            }
            .instrument(info_span!("walredo-stdin"));

            #[derive(Debug, thiserror::Error)]
            enum StdoutTaskError {
                #[error("read failed: {0}")]
                ReadFailed(std::io::Error),
                #[error("external process stdout was closed")]
                StdoutClosed,
                #[error("reading the page timed out")]
                ReadTimeout,
            }

            let stdout_task = async {
                // TODO: do these pages are put it in a cache? if not, could use a larger buffer
                let mut result_rx = result_rx;
                let mut page_buf = BytesMut::with_capacity(8192);

                while let Some((completion, timeout_at)) = result_rx.recv().await {
                    let read_page = async {
                        loop {
                            let read = stdout
                                .read_buf(&mut page_buf)
                                .await
                                .map_err(StdoutTaskError::ReadFailed)?;
                            if read == 0 {
                                return Err(StdoutTaskError::StdoutClosed);
                            }
                            if page_buf.remaining() < 8192 {
                                continue;
                            }
                            let page = page_buf.split().freeze();
                            return Ok(page);
                        }
                    };

                    let res = tokio::time::timeout_at(timeout_at, read_page)
                        .await
                        .map_err(|_elapsed| StdoutTaskError::ReadTimeout)
                        .and_then(|x| x);

                    match res {
                        Ok(page) => {
                            // we don't care about the result, because the caller could be gone
                            drop(completion.send(Ok(page)));
                            page_buf.reserve(8192);
                        }
                        Err(StdoutTaskError::ReadFailed(e)) => {
                            drop(
                                completion
                                    .send(Err(e).context("Failed to read response from wal-redo")),
                            );
                            return Err("failed to read from wal-redo stdout");
                        }
                        Err(StdoutTaskError::StdoutClosed) => {
                            drop(
                                completion
                                    .send(Err(anyhow::anyhow!("wal-redo process closed stdout"))),
                            );
                            return Err("failed to read from wal-redo: closed stdout");
                        }
                        Err(StdoutTaskError::ReadTimeout) => {
                            drop(completion.send(Err(anyhow::anyhow!(
                                "Timed out while waiting for the page"
                            ))));
                            return Err("reading page timed out");
                        }
                    }
                }
                // in a graceful shutdown, this needs to be an Err to take down the stderr task as
                // well.
                Err::<(), _>("stdout: all requests processed, ready for shutdown")
            }
            .instrument(info_span!("walredo-stdout"));

            let stderr_task = async {
                let mut buffer = Vec::new();

                loop {
                    buffer.clear();
                    match stderr.read_until(b'\n', &mut buffer).await {
                        Ok(0) => return Err::<(), _>("stderr: closed"),
                        Ok(read) => {
                            let message = String::from_utf8_lossy(&buffer[..read]);
                            error!("wal-redo-process: {}", message.trim());
                        }
                        Err(e) => {
                            error!("reading stderr failed: {e}");
                            return Err("stderr: read failed");
                        }
                    }
                }
            };

            async {
                // ignore the result, it is always Err from one of the tasks, upon which we stop
                // advancing the others
                let reason = tokio::try_join!(stdin_task, stdout_task, stderr_task);

                debug!("wal-redo process tasks exited: {reason:?}");

                // dont care if the child has already exited
                drop(child.start_kill());

                match child.wait().await {
                    Ok(status) => {
                        if status.success() {
                            debug!(?status, "wal-redo process exited successfully");
                        } else {
                            warn!(?status, "wal-redo process did not exit successfully");
                        }
                    }
                    Err(e) => {
                        error!("failed to wait for child process to exit: {e}");
                    }
                }
            }
            .instrument(info_span!("walredo", pid, %tenant_id))
            .await
        }

        info!(tenant_id = %tenant_id, "wal-redo task exiting");
        Ok(())
    };

    (Handle { tx }, ipc)
}

async fn launch_walredo(
    conf: &PageServerConf,
    tenant_id: TenantId,
    pg_version: u32,
) -> anyhow::Result<tokio::process::Child> {
    let datadir = path_with_suffix_extension(
        conf.tenant_path(&tenant_id).join("wal-redo-datadir"),
        TEMP_FILE_SUFFIX,
    );

    info!("removing existing data directory: {}", datadir.display());

    match tokio::fs::remove_dir_all(&datadir).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        other => other.with_context(|| {
            format!(
                "Failed to cleanup existing wal-redo-datadir at {}",
                datadir.display()
            )
        })?,
    }

    let pg_bin_dir_path = conf.pg_bin_dir(pg_version)?;
    let pg_lib_dir_path = conf.pg_lib_dir(pg_version)?;

    info!("running initdb in {}", datadir.display());

    let initdb = tokio::process::Command::new(pg_bin_dir_path.join("initdb"))
        .arg("-D")
        .arg(&datadir)
        .arg("-N")
        .env_clear()
        .env("LD_LIBRARY_PATH", &pg_lib_dir_path)
        .env("DYLD_LIBRARY_PATH", &pg_lib_dir_path)
        .close_fds()
        .output()
        .await
        .context("Failed to execute initdb for wal-redo")?;

    anyhow::ensure!(
        initdb.status.success(),
        "initdb failed\nstdout: {}\nstderr:\n {}",
        String::from_utf8_lossy(&initdb.stdout),
        String::from_utf8_lossy(&initdb.stderr)
    );

    info!("starting walredo process");

    tokio::process::Command::new(pg_bin_dir_path.join("postgres"))
        .arg("--wal-redo")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .env_clear()
        .env("LD_LIBRARY_PATH", &pg_lib_dir_path)
        .env("DYLD_LIBRARY_PATH", &pg_lib_dir_path)
        .env("PGDATA", &datadir)
        .close_fds()
        // best effort is probably good enough for us
        .kill_on_drop(true)
        .spawn()
        .context("postgres --wal-redo command failed to start")
}

type Payload = (Request, tokio::sync::oneshot::Sender<anyhow::Result<Bytes>>);

/// WAL Redo request
struct Request {
    target: BufferTag,
    base_img: Option<Bytes>,
    records: Arc<[(Lsn, NeonWalRecord)]>,
    records_range: SliceRange,
    timeout: std::time::Duration,
}

#[derive(Clone)]
struct Handle {
    tx: tokio::sync::mpsc::Sender<Payload>,
}

impl Handle {
    async fn request_redo(&self, request: Request) -> anyhow::Result<Bytes> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        self.tx
            .send((request, result_tx))
            .await
            .map_err(|_| anyhow::anyhow!("Failed to communicate with the walredo task"))?;

        let res = result_rx
            .await
            .context("Failed to get a WAL Redo'd page back")
            .and_then(|x| x);

        res
    }
}

enum SliceRange {
    InclusiveExclusive(std::ops::Range<usize>),
    RangeFrom(std::ops::RangeFrom<usize>),
}

impl From<std::ops::Range<usize>> for SliceRange {
    fn from(r: std::ops::Range<usize>) -> Self {
        SliceRange::InclusiveExclusive(r)
    }
}

impl From<std::ops::RangeFrom<usize>> for SliceRange {
    fn from(r: std::ops::RangeFrom<usize>) -> Self {
        SliceRange::RangeFrom(r)
    }
}

impl SliceRange {
    fn sub_slice<'a, T>(&self, full_slice: &'a [T]) -> &'a [T] {
        match self {
            SliceRange::InclusiveExclusive(r) => &full_slice[r.start..r.end],
            SliceRange::RangeFrom(r) => &full_slice[r.start..],
        }
    }
}

/// Conceptually `Vec<Bytes>` masquerading as `bytes::Buf`.
///
/// Used to build vectorized writes, in case we have a base page.
///
/// Adapted from https://github.com/tokio-rs/bytes/pull/371
#[derive(Default)]
struct BufQueue {
    bufs: std::collections::VecDeque<Bytes>,
    remaining: usize,
}

impl BufQueue {
    fn push(&mut self, buffer: Bytes) {
        let rem = buffer.remaining();
        if rem != 0 {
            self.bufs.push_back(buffer);
            self.remaining = self.remaining.checked_add(rem).expect("remaining overflow");
        }
    }

    fn clear(&mut self) {
        self.bufs.clear();
        self.remaining = 0;
    }
}

impl bytes::Buf for BufQueue {
    fn remaining(&self) -> usize {
        self.remaining
    }

    fn chunk(&self) -> &[u8] {
        match self.bufs.front() {
            Some(b) => b.chunk(),
            None => &[],
        }
    }

    fn chunks_vectored<'a>(&'a self, mut dst: &mut [std::io::IoSlice<'a>]) -> usize {
        let mut n = 0;

        for b in &self.bufs {
            if dst.is_empty() {
                break;
            }
            let next = b.chunks_vectored(dst);
            dst = &mut dst[next..];
            n += next;
        }

        n
    }

    fn advance(&mut self, mut cnt: usize) {
        while cnt != 0 {
            let front = self.bufs.front_mut().expect("mut not be empty");
            let rem = front.remaining();
            let advance = std::cmp::min(cnt, rem);
            front.advance(advance);
            if rem == advance {
                self.bufs.pop_front().unwrap();
            }
            cnt -= advance;
            self.remaining -= advance;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PostgresRedoManager, WalRedoManager};
    use crate::repository::Key;
    use crate::{config::PageServerConf, walrecord::NeonWalRecord};
    use bytes::Bytes;
    use std::str::FromStr;
    use utils::{id::TenantId, lsn::Lsn};

    #[test]
    fn short_v14_redo() {
        // prettier than embedding the 8192 bytes here though most of it are zeroes
        // PRE-MERGE: zstd would cut it to 263 bytes, consider in review?
        let expected = std::fs::read("fixtures/short_v14_redo.page").unwrap();

        let h = RedoHarness::new().unwrap();
        let page = h
            .manager
            .request_redo(
                Key {
                    field1: 0,
                    field2: 1663,
                    field3: 13010,
                    field4: 1259,
                    field5: 0,
                    field6: 0,
                },
                Lsn::from_str("0/16E2408").unwrap(),
                None,
                short_records(),
                14,
            )
            .unwrap();

        assert_eq!(&expected, &*page);
    }

    #[test]
    fn short_v14_fails_for_wrong_key_but_returns_zero_page() {
        let h = RedoHarness::new().unwrap();

        let page = h
            .manager
            .request_redo(
                Key {
                    field1: 0,
                    field2: 1663,
                    // key should be 13010
                    field3: 13130,
                    field4: 1259,
                    field5: 0,
                    field6: 0,
                },
                Lsn::from_str("0/16E2408").unwrap(),
                None,
                short_records(),
                14,
            )
            .unwrap();

        // TODO: there will be some stderr printout, which is forwarded to tracing that could
        // perhaps be captured as long as it's in the same thread.
        assert_eq!(page, crate::ZERO_PAGE);
    }

    #[allow(clippy::octal_escapes)]
    fn short_records() -> Vec<(Lsn, NeonWalRecord)> {
        vec![
            (
                Lsn::from_str("0/16A9388").unwrap(),
                NeonWalRecord::Postgres {
                    will_init: true,
                    rec: Bytes::from_static(b"j\x03\0\0\0\x04\0\0\xe8\x7fj\x01\0\0\0\0\0\n\0\0\xd0\x16\x13Y\0\x10\0\04\x03\xd4\0\x05\x7f\x06\0\0\xd22\0\0\xeb\x04\0\0\0\0\0\0\xff\x03\0\0\0\0\x80\xeca\x01\0\0\x01\0\xd4\0\xa0\x1d\0 \x04 \0\0\0\0/\0\x01\0\xa0\x9dX\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0.\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\00\x9f\x9a\x01P\x9e\xb2\x01\0\x04\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x02\0!\0\x01\x08 \xff\xff\xff?\0\0\0\0\0\0@\0\0another_table\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x98\x08\0\0\x02@\0\0\0\0\0\0\n\0\0\0\x02\0\0\0\0@\0\0\0\0\0\0\0\0\0\0\0\0\x80\xbf\0\0\0\0\0\0\0\0\0\0pr\x01\0\0\0\0\0\0\0\0\x01d\0\0\0\0\0\0\x04\0\0\x01\0\0\0\0\0\0\0\x0c\x02\0\0\0\0\0\0\0\0\0\0\0\0\0\0/\0!\x80\x03+ \xff\xff\xff\x7f\0\0\0\0\0\xdf\x04\0\0pg_type\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0b\0\0\0G\0\0\0\0\0\0\0\n\0\0\0\x02\0\0\0\0\0\0\0\0\0\0\0\x0e\0\0\0\0@\x16D\x0e\0\0\0K\x10\0\0\x01\0pr \0\0\0\0\0\0\0\0\x01n\0\0\0\0\0\xd6\x02\0\0\x01\0\0\0[\x01\0\0\0\0\0\0\0\t\x04\0\0\x02\0\0\0\x01\0\0\0\n\0\0\0\n\0\0\0\x7f\0\0\0\0\0\0\0\n\0\0\0\x02\0\0\0\0\0\0C\x01\0\0\x15\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0.\0!\x80\x03+ \xff\xff\xff\x7f\0\0\0\0\0;\n\0\0pg_statistic\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x0b\0\0\0\xfd.\0\0\0\0\0\0\n\0\0\0\x02\0\0\0;\n\0\0\0\0\0\0\x13\0\0\0\0\0\xcbC\x13\0\0\0\x18\x0b\0\0\x01\0pr\x1f\0\0\0\0\0\0\0\0\x01n\0\0\0\0\0\xd6\x02\0\0\x01\0\0\0C\x01\0\0\0\0\0\0\0\t\x04\0\0\x01\0\0\0\x01\0\0\0\n\0\0\0\n\0\0\0\x7f\0\0\0\0\0\0\x02\0\x01")
                }
            ),
            (
                Lsn::from_str("0/16D4080").unwrap(),
                NeonWalRecord::Postgres {
                    will_init: false,
                    rec: Bytes::from_static(b"\xbc\0\0\0\0\0\0\0h?m\x01\0\0\0\0p\n\0\09\x08\xa3\xea\0 \x8c\0\x7f\x06\0\0\xd22\0\0\xeb\x04\0\0\0\0\0\0\xff\x02\0@\0\0another_table\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x98\x08\0\0\x02@\0\0\0\0\0\0\n\0\0\0\x02\0\0\0\0@\0\0\0\0\0\0\x05\0\0\0\0@zD\x05\0\0\0\0\0\0\0\0\0pr\x01\0\0\0\0\0\0\0\0\x01d\0\0\0\0\0\0\x04\0\0\x01\0\0\0\x02\0")
                }
            )
        ]
    }

    struct RedoHarness {
        // underscored because unused, except for removal at drop
        _repo_dir: tempfile::TempDir,
        manager: PostgresRedoManager,
        // FIXME: this needs a own tokio reactor to use the same api
    }

    impl RedoHarness {
        fn new() -> anyhow::Result<Self> {
            let repo_dir = tempfile::tempdir()?;
            let conf = PageServerConf::dummy_conf(repo_dir.path().to_path_buf());
            let conf = Box::leak(Box::new(conf));
            let tenant_id = TenantId::generate();

            let manager = PostgresRedoManager::new(conf, tenant_id);

            Ok(RedoHarness {
                _repo_dir: repo_dir,
                manager,
            })
        }
    }
}
