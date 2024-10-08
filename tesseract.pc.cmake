prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=@CMAKE_INSTALL_PREFIX@
libdir=@CMAKE_INSTALL_FULL_LIBDIR@
includedir=@CMAKE_INSTALL_FULL_INCLUDEDIR@

Name: @tesseract_NAME@
Description: An OCR Engine that was developed at HP Labs between 1985 and 1995... and now at Google.
URL: https://github.com/tesseract-ocr/tesseract
Version: @tesseract_VERSION@
Requires.private: lept
Libs: -L${libdir} -l@tesseract_OUTPUT_NAME@ @libarchive_LIBS@ @libcurl_LIBS@ @TENSORFLOW_LIBS@
Libs.private:
Cflags: -I${includedir}
