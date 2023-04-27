AC_PREREQ([2.65])
AC_INIT([nestgpu], [1.8.2], [])

# Config initialization
AC_CONFIG_SRCDIR([m4/ct_check_cuda.m4])
AC_CONFIG_SRCDIR([m4/pypath.m4])
#, [m4/ct_check_cuda_sdk.m4])
AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_HEADER(config.h)
AC_CONFIG_MACRO_DIRS([m4])

AC_ENABLE_SHARED
AC_DISABLE_STATIC
LT_INIT

# automake initialization
AM_INIT_AUTOMAKE

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AM_PROG_CC_C_O

# AC_CANONICAL_HOST is needed to access the 'host_os' variable    
AC_CANONICAL_HOST

build_linux=no
build_windows=no
build_mac=no

# Detect the target system
case "${host_os}" in
    linux*)
        build_linux=yes
        ;;
    cygwin*|mingw*)
        build_windows=yes
        ;;
    darwin*)
        build_mac=yes
        ;;
    *)
        AC_MSG_ERROR(["OS $host_os is not supported"])
        ;;
esac

# Pass the conditionals to automake
AM_CONDITIONAL([LINUX], [test "$build_linux" = "yes"])
AM_CONDITIONAL([WINDOWS], [test "$build_windows" = "yes"])
AM_CONDITIONAL([OSX], [test "$build_mac" = "yes"])

# Search nvcc compiler
AC_PATH_PROG(NVCC, nvcc, "no")
AS_IF([test "x$NVCC" = "xno"],[AC_MSG_ERROR([NVCC compiler not found!])])
# Make variable $NVCC available in Makefile.am
AC_SUBST(NVCC)

# check for OpenMP
AC_OPENMP

#check for Python
AM_PATH_PYTHON([3.8])
#adl_CHECK_PYTHON
#AC_MSG_ERROR([$PYTHONINC])


# Checks for libraries.
AC_CHECK_LIB([m], [pow])
#AC_CHECK_LIB([just_a_test], [pow], [], [AC_MSG_ERROR([pow was not found in libjust_a_test])])

# Checks for header files.
AC_CHECK_HEADERS([limits.h stdlib.h string.h])

# check for CUDA
#CT_CHECK_CUDA_API

# check for CUDA SDK
#CT_CHECK_CUDA_SDK

GPU_ARCH=sm_35
AC_ARG_WITH(gpu-arch, [AC_HELP_STRING([--with-gpu-arch=GPU architecture],
[Specify the GPU compute capability [default=sm_35]])],
[GPU_ARCH=$withval])
AC_SUBST(GPU_ARCH)
AC_MSG_NOTICE(GPU architecture: ${GPU_ARCH})

dnl Check for MPI
dnl This check will set the MPICC and MPICXX variables to the MPI compiler ones
dnl if the library is found, or to the regular compilers if not.
AC_ARG_WITH(mpi, [AC_HELP_STRING([--with-mpi],
[enable MPI support [default=yes]])],
[case "${withval}" in
yes|no) with_mpi=$withval;;
*)
AC_MSG_ERROR(bad value ${withval} for --with-mpi);;
esac],
[with_mpi=yes])
if test "x$with_mpi" = "xyes"; then
ACX_MPI([], [AC_MSG_ERROR(m4_normalize([
could not find MPI library. You should either install it,
or use the option --with-mpi=no
to build without MPI-related functions]))])
AC_DEFINE(HAVE_MPI)
MPICC="mpicc"
MPICXX="mpicxx"
MPI_CXXLIBS=`mpicxx --showme:link`
MPI_CXXFLAGS=`mpicxx --showme:compile`
AC_SUBST(MPI_CXXLIBS)
AC_SUBST(MPI_CXXFLAGS)
else
MPICC="$CC"
MPICXX="$CXX"
AC_SUBST(MPICC)
AC_SUBST(MPICXX)
fi
AM_CONDITIONAL([WE_HAVE_MPI],[test "x$with_mpi" = "xyes"])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([pow])

AC_CONFIG_FILES([Makefile])


AC_OUTPUT
_AC_SRCDIRS(["."])
export ac_top_srcdir
$ac_top_srcdir/patch.sh
