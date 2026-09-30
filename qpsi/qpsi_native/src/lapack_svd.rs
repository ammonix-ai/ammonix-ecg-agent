//! LAPACK dgesdd SVD via runtime DLL loading (no external crate).
//!
//! On Windows: uses LoadLibraryW / GetProcAddress directly.
//! On Linux/macOS: uses dlopen / dlsym.
//! Falls back to nalgebra when the DLL is not found.
//!
//! Thread-safe: each call allocates its own workspace buffers.
//! MKL_NUM_THREADS=1 is set to avoid MKL/rayon contention.

use std::sync::OnceLock;
use pyo3::prelude::PyAnyMethods;

// ---------------------------------------------------------------------------
// FFI type for LAPACK dgesdd_ (Fortran calling convention, all by pointer)
// ---------------------------------------------------------------------------

type DgesddFn = unsafe extern "C" fn(
    jobz: *const u8,
    m: *const i32,
    n: *const i32,
    a: *mut f64,
    lda: *const i32,
    s: *mut f64,
    u: *mut f64,
    ldu: *const i32,
    vt: *mut f64,
    ldvt: *const i32,
    work: *mut f64,
    lwork: *const i32,
    iwork: *mut i32,
    info: *mut i32,
);

struct LapackHandle {
    #[cfg(windows)]
    _handle: *mut std::ffi::c_void,
    #[cfg(not(windows))]
    _handle: *mut std::ffi::c_void,
    dgesdd: DgesddFn,
}

unsafe impl Send for LapackHandle {}
unsafe impl Sync for LapackHandle {}

static LAPACK: OnceLock<Option<LapackHandle>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Platform-specific DLL loading
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;

    extern "system" {
        fn LoadLibraryW(lpFileName: *const u16) -> *mut c_void;
        fn GetProcAddress(hModule: *mut c_void, lpProcName: *const u8) -> *mut c_void;
        fn SetDllDirectoryW(lpPathName: *const u16) -> i32;
    }

    fn to_wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub fn add_dll_directory(dir: &str) {
        let wide = to_wide(dir);
        unsafe { SetDllDirectoryW(wide.as_ptr()); }
    }

    pub fn load_library(path: &str) -> *mut c_void {
        let wide = to_wide(path);
        unsafe { LoadLibraryW(wide.as_ptr()) }
    }

    pub fn get_proc(handle: *mut c_void, name: &[u8]) -> *mut c_void {
        unsafe { GetProcAddress(handle, name.as_ptr()) }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::ffi::{c_void, CString};

    extern "C" {
        fn dlopen(filename: *const i8, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    }

    const RTLD_NOW: i32 = 2;

    pub fn add_dll_directory(_dir: &str) {}

    pub fn load_library(path: &str) -> *mut c_void {
        let c_path = CString::new(path).unwrap();
        unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) }
    }

    pub fn get_proc(handle: *mut c_void, name: &[u8]) -> *mut c_void {
        let c_name = CString::new(name).unwrap();
        unsafe { dlsym(handle, c_name.as_ptr()) }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct SvdResult {
    pub u_col: Vec<f64>,
    pub s: Vec<f64>,
    pub vt_col: Vec<f64>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub fn init(py: pyo3::Python<'_>) {
    LAPACK.get_or_init(|| {
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OMP_NUM_THREADS", "1");

        // Strategy 1: explicit env var
        if let Ok(path) = std::env::var("QPSI_LAPACK_PATH") {
            if let Some(h) = try_load(&path) {
                eprintln!("[qpsi_native] LAPACK loaded from QPSI_LAPACK_PATH: {}", path);
                return Some(h);
            }
        }

        // Strategy 2: discover via Python sys.prefix
        if let Some(path) = discover_via_python(py) {
            if let Some(h) = try_load(&path) {
                eprintln!("[qpsi_native] LAPACK loaded from pixi env: {}", path);
                return Some(h);
            }
        }

        // Strategy 3: system path
        #[cfg(windows)]
        let names = &["liblapack.dll"];
        #[cfg(target_os = "linux")]
        let names = &["liblapack.so", "liblapack.so.3"];
        #[cfg(target_os = "macos")]
        let names = &["liblapack.dylib"];

        for name in names {
            if let Some(h) = try_load(name) {
                eprintln!("[qpsi_native] LAPACK loaded: {}", name);
                return Some(h);
            }
        }

        eprintln!("[qpsi_native] LAPACK DLL not found; falling back to nalgebra SVD");
        None
    });
}

pub fn is_available() -> bool {
    matches!(LAPACK.get(), Some(Some(_)))
}

pub fn svd_decompose(a_row: &[f64], m: usize, n: usize) -> SvdResult {
    if let Some(Some(h)) = LAPACK.get() {
        svd_via_lapack(h, a_row, m, n)
    } else {
        svd_via_nalgebra(a_row, m, n)
    }
}

// ---------------------------------------------------------------------------
// LAPACK dgesdd path
// ---------------------------------------------------------------------------

fn svd_via_lapack(h: &LapackHandle, a_row: &[f64], m: usize, n: usize) -> SvdResult {
    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k = m.min(n);
    let k_i32 = k as i32;

    // Row-major → column-major
    let mut a_col = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a_col[j * m + i] = a_row[i * n + j];
        }
    }

    let mut s = vec![0.0f64; k];
    let mut u = vec![0.0f64; m * k];
    let mut vt = vec![0.0f64; k * n];
    let mut iwork = vec![0i32; 8 * k];
    let mut info: i32 = 0;
    let lda = m_i32;
    let ldu = m_i32;
    let ldvt = k_i32;
    let jobz: u8 = b'S';

    // Workspace query
    let mut work_query = [0.0f64; 1];
    let lwork_query: i32 = -1;

    unsafe {
        (h.dgesdd)(
            &jobz, &m_i32, &n_i32,
            a_col.as_mut_ptr(), &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(), &ldu,
            vt.as_mut_ptr(), &ldvt,
            work_query.as_mut_ptr(), &lwork_query,
            iwork.as_mut_ptr(), &mut info,
        );
    }

    let optimal_lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; optimal_lwork.max(1) as usize];

    // Actual SVD
    unsafe {
        (h.dgesdd)(
            &jobz, &m_i32, &n_i32,
            a_col.as_mut_ptr(), &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(), &ldu,
            vt.as_mut_ptr(), &ldvt,
            work.as_mut_ptr(), &optimal_lwork,
            iwork.as_mut_ptr(), &mut info,
        );
    }

    if info != 0 {
        eprintln!("[qpsi_native] dgesdd failed (info={}), falling back to nalgebra", info);
        return svd_via_nalgebra(a_row, m, n);
    }

    SvdResult { u_col: u, s, vt_col: vt, m, n, k }
}

// ---------------------------------------------------------------------------
// nalgebra fallback
// ---------------------------------------------------------------------------

fn svd_via_nalgebra(a_row: &[f64], m: usize, n: usize) -> SvdResult {
    let k = m.min(n);
    let a_mat = nalgebra::DMatrix::from_row_slice(m, n, a_row);
    let svd = nalgebra::SVD::new(a_mat, true, true);

    let u_na = svd.u.as_ref().unwrap();
    let vt_na = svd.v_t.as_ref().unwrap();
    let s_na = &svd.singular_values;

    let mut u_col = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            u_col[j * m + i] = u_na[(i, j)];
        }
    }

    let s: Vec<f64> = s_na.iter().copied().collect();

    let mut vt_col = vec![0.0; k * n];
    for i in 0..k {
        for j in 0..n {
            vt_col[j * k + i] = vt_na[(i, j)];
        }
    }

    SvdResult { u_col, s, vt_col, m, n, k }
}

// ---------------------------------------------------------------------------
// DLL discovery
// ---------------------------------------------------------------------------

fn discover_via_python(py: pyo3::Python<'_>) -> Option<String> {
    let sys = py.import("sys").ok()?;
    let prefix: String = sys.getattr("prefix").ok()?.extract().ok()?;
    let prefix = prefix.replace('\\', "/");

    #[cfg(windows)]
    {
        let path = format!("{}/Library/bin/liblapack.dll", prefix);
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }

    #[cfg(not(windows))]
    {
        let path = format!("{}/lib/liblapack.so", prefix);
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }

    None
}

fn try_load(path: &str) -> Option<LapackHandle> {
    // Add DLL directory to search path for MKL dependencies
    if let Some(dir) = std::path::Path::new(path).parent() {
        if dir.exists() {
            platform::add_dll_directory(&dir.to_string_lossy());
            // Also prepend to PATH as fallback
            let current = std::env::var("PATH").unwrap_or_default();
            let dir_str = dir.to_string_lossy();
            if !current.contains(&*dir_str) {
                std::env::set_var("PATH", format!("{};{}", dir_str, current));
            }
        }
    }

    let handle = platform::load_library(path);
    if handle.is_null() {
        return None;
    }

    // Try Fortran-mangled names
    let dgesdd_ptr = platform::get_proc(handle, b"dgesdd_\0");
    let dgesdd_ptr = if dgesdd_ptr.is_null() {
        platform::get_proc(handle, b"dgesdd\0")
    } else {
        dgesdd_ptr
    };
    let dgesdd_ptr = if dgesdd_ptr.is_null() {
        platform::get_proc(handle, b"DGESDD\0")
    } else {
        dgesdd_ptr
    };

    if dgesdd_ptr.is_null() {
        return None;
    }

    let dgesdd: DgesddFn = unsafe { std::mem::transmute(dgesdd_ptr) };

    Some(LapackHandle {
        _handle: handle,
        dgesdd,
    })
}
