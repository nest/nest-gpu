AC_DEFUN([adl_CHECK_PYTHON],
 [AM_PATH_PYTHON([2.0])
  AC_CACHE_CHECK([for $am_display_PYTHON includes directory],
    [adl_cv_python_inc],
    [adl_cv_python_inc=`$PYTHON -c "from distutils import sysconfig; print sysconfig.get_python_inc()" 2>/dev/null`])
  AC_SUBST([PYTHONINC], [$adl_cv_python_inc])])
