;; makefile.lsp -- builds makefiles for various machine types

(setf system-types '(rs6k next pmax sparc sgi linux))

(if (not (boundp 'system-type)) (setf system-type nil))
(if (not (boundp 'target-file)) (setf target-file "ny"))

(format t "System types: ~A~%" system-types)
(format t "Current type: ~A~%" system-type)
(format t "Current target: ~A~%" target-file)
(format t "~%Instructions: (run from top nyquist directory)~%")
(format t "Choose a system from the list above by typing:~%")
(format t "\t(setf system-type '<a system type>)~%")
(format t "Override the executable name or location by:~%")
(format t "\t(setf target-file \"unix-path-name/ny\")~%")
(format t "To build the Makefile, type:~%")
(format t "\t(makefile)~%")
(format t "To make Makefiles for all system types, type:~%")
(format t "\t(makeall)~%")
(format t "To make sndfn.wcl and sndfn.cl, type:~%")
(format t "\t(commandline)~%")

;(format t "To build the Makesrc file, type:~%")
;(format t "\t(makesrc)~%")
;(format t
;"Note: Makesrc is used to update sources from other directories.
;    It isn't necessary if you got the sources from the normal
;    .tar file release of Nyquist
;")


(setf xlfiles '("extern" "xldmem" 
  "xlbfun" "xlcont" "xldbug" "xleval"
  "xlfio" "xlftab" "xlglob" "xlimage" "xlinit" "xlio" "xlisp"
  "xljump" "xllist" "xlmath" "xlobj" "xlpp" "xlprin" "xlread"
  "xlstr" "xlsubr" "xlsym" "xlsys" "path"))

(setf xlfiles-h '("osdefs" "osptrs" "xldmem" "xlisp" "extern"))

(setf xlfiles-lsp '("xlinit" "misc" "evalenv" "printrec"))

(setf nyqfiles '("debug" "falloc" "add" "local" "convolve"
  "downsample" "handlers" "multiread" "multiseq" "samples"
  "seqext" "seqinterf" "sndread" "sndseq" "yin"
  "sndwritepa" "sndmax" "sound" "stats" "compose" "inverse"
  "resamp" "resampv" "ffilterkit" "avg" "fft" "sndfail"
  "lpanal"))

(setf stksrcfiles '("Clarinet" "Delay" "DelayL" "Envelope" "Filter" 
  "Instrmnt" "Noise" "OneZero" "ReedTabl" "Saxofony" "Stk"
  "WaveLoop" "WvIn"))

(setf stkfiles '("stkinit" "instr"))

(setf fftfiles '("fftn"))

(setf pafiles '("pa_common/pa_front" "pa_unix/pa_unix_hostapis" 
                "pa_unix_oss/pa_unix_oss" "pa_unix/pa_unix_util"
                "pa_common/pa_cpuload" "pa_common/pa_allocation"
                "pa_common/pa_stream" "pa_common/pa_converters"
                "pa_common/pa_process" "pa_common/pa_dither"
                "pa_common/pa_trace"))

; note: audio<sys> and snd<sys> will be prepended to this list, e.g.
;   the strings "audiooss" and "sndlinux" will be added for linux systems
;
(defun init-sndfiles ()
  (setf sndfiles '("ieeecvt" "snd" "sndcvt" "sndio" "sndheader"))
  (setf sndfiles-lsp '("snd")))

(init-sndfiles)

(setf depends-exceptions '(
   ("nyqsrc/handlers" "") 
   ("nyqsrc/sndfail" "")
   ("nyqsrc/local" "xlisp/xlisp.h nyqsrc/sound.h")
   ("nyqsrc/stats" "nyqsrc/sound.h nyqsrc/falloc.h nyqsrc/cque.h")
   ("snd/sndcvt" "snd/snd.h")
   ("snd/sndio" "snd/snd.h")
   ("snd/audiors6k" "snd/snd.h")
   ("snd/audionext" "snd/snd.h")
   ("snd/audiosgi" "snd/snd.h")
   ("snd/audiopmax" "snd/snd.h")
   ("snd/audiosparc" "snd/snd.h")
   ("snd/audiolinux" "snd/snd.h")
   ("snd/audiooss" "snd/snd.h")
   ("nyqsrc/sndwritepa" "nyqsrc/sndwrite.h")
   ("nyqsrc/sndfnint" "")  ; sparc needs explicit rule for sndfnint.o
   ("nyqsrc/seqfnint" "")  ; ditto for seqfnint.o
))

(setf nyqfiles-lsp '("init" "nyquist" "seqmidi" "seq" "makefile" "update" "transfiles" "examples" "nyinit"))

(setf system-types-as-strings (mapcar #'string-downcase 
                (mapcar #'symbol-name system-types)))
(setf nyqfiles-lsp (append nyqfiles-lsp system-types-as-strings))

(setf nyqfiles-h '("localdefs" "localptrs" "seqdecls" "cque" "switches"))

(setf intfiles '("sndfnint" "seqfnint"))

(setf extrafiles nil)
;(dolist (m system-types)
;        (push (strcat "Makefile." 
;                      (string-downcase (symbol-name m)))
;              extrafiles))
(push "export" extrafiles)
(push "README" extrafiles)
(push "howtorelease.doc" extrafiles)

(setf cmtfiles '("cext" "cleanup" "cmdline" "cmtcmd" 
  "moxc" "mem" "midifile" "midifns" "record"
  "seq" "seqmread" "seqmwrite" "seqread" "seqwrite" "tempomap"
  "timebase" "userio")) ; "midimgr" - removed by RBD

(setf cmtfiles-h '("mfmidi" "midicode" "midierr" "musiprog"
  "pitch"  "swlogic" "hash" "hashrout" "io" "midibuff"))


(defun insert-separator (pre sep lis)
  (mapcar #'(lambda (pair) 
              (cond ((= (length pair) 2)
                     (strcat pre (car pair) sep (cadr pair) ".h"))
                    (t
                     (strcat (car pair) pre (cadr pair) sep (caddr pair) ".h"))))
          lis))

;; COMMAND-PREFIX -- insert prefix before each file
;;
(defun command-prefix (prefix lis)
  (mapcar #'(lambda (item) (list prefix item))
          lis))

;; COMMAND-FILELIST -- build files for command line
;;
(defun command-filelist (prefix separator)
  (let ()
    (setf filelist '(("snd" "snd")))
    (setf filelist (append filelist (command-prefix "nyqsrc" nyqsrcfiles)))
    (setf filelist (append filelist '(("~" "nyqsrc" "sndheader"))))
    (setf filelist (append filelist (command-prefix "tran" transfiles)))
    (cons (strcat prefix "nyqsrc" separator "sndfnint") 
          (insert-separator prefix separator filelist))))


;; COMMANDLINE -- build sndfn.cl and sndfn.wcl for mac and windows
;; versions of intgen; the files will be written to currentn directory
;;
(defun commandline ()
  (load "misc/transfiles") ;; get current versions of transfiles and nyqsrcfiles
  (let (filelist outf)
    (setf filelist (command-filelist "" "\\"))
    (setf outf (open "sndfn.wcl" :direction :output))
    (write-file-list outf filelist #\ )
    (close outf)
    ; now do the mac
    (setf filelist (command-filelist ":" ":"))
    (setf outf (open "sndfn.cl" :direction :output))
    (write-file-list outf filelist #\ )
    (close outf)
    ))

;; MAKEALL - makes all makefiles and copies them to nyqsrc
;; 
;; run this in nyquist/src
;; 
(defun makeall ()
;  (makesrc)
;  (system "cp -p Makesrc nyqsrc")
  (dolist (m system-types)
      (setf system-type m)
      (setf m (string-downcase (symbol-name m)))
      (init-sndfiles)
      (makefile)))

;; MAKE-AUDIO-NAME -- (strcat "audio" system-name)
(defun make-audio-name (system-name)
  (cond ((eq system-type 'linux)
         "audiooss")
        (t (strcat "audio" system-name))))

;; MAKE-SND-NAME -- (strcat "audio" system-name)
(defun make-snd-name (system-name)
   (strcat "snd" system-name))


;; MAKEFILE - creates a Makefile from a list of sources
;;
;; reads sources from nyqfiles.txt
(defun makefile ()
  (let (system-name outf outf-name)
    (load "misc/transfiles.lsp") ; just to make sure we're current
    (while (null system-type)
       (format t "Write Makefile for what system?  One of:~A~%" system-types)
       (setf system-type (read))
       (cond ((not (member system-type system-types))
          (format t "Unknown system type.~%")
          (setf system-type nil))))
    (setf system-name (string-downcase
      (symbol-name system-type)))
    (setf outf-name (strcat "sys/unix/" system-name "/Makefile"))
    (format t "Opening for output: ~A\n" outf-name)
    (setf outf (open outf-name :direction :output))
    (cond ((null outf)
           (error "could not open output file" outf-name)))
    (setf sndfiles (cons (make-audio-name system-name)
             (cons (make-snd-name system-name) sndfiles)))
    (format outf 
     "#
# Makefile for Nyquist, SYSTEM-TYPE is ~A
# run make in the top-level Nyquist directory to compile Nyquist
#
# NOTE: this file is machine-generated.  DO NOT EDIT!
#   Instead, modify makefile.lsp and regenerate the makefile.
#   Ports and bug fixes are welcome - please mail them to 
#   dannenberg@cs.cmu.edu.  Thanks.
#

# This is the resulting executable (normally \"ny\"):
NY = ~A

EVERYTHING = $(NY) runtime/system.lsp

CURRENT = $(EVERYTHING)

current: $(CURRENT)

# Standard list of includes (common to all unix versions)
INCL = -Inyqsrc -Itran -Ixlisp -Isys/unix -Icmt -Isnd -Ifft \\
   -Inyqstk/include -Inyqstk -Iportaudio/pa_common

# system dependent stuff for ~A:
~A

INTGEN = misc/intgen

# Object files for Nyquist:
" system-type target-file system-name (system-defs))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    (object-files outf)
    (format outf "# Sound functions to add to xlisp~%")
    (nyqheaders outf)
    (cmtheaders outf)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    (format outf "

$(NY): $(OBJECTS)
\t$(LN) $(OBJECTS) $(LFLAGS) -o $(NY)

# copy appropriate system.lsp and make it read-only;
# changes should be made to sys/unix/<system>/system.lsp
runtime/system.lsp: sys/unix/~A/system.lsp
\tchmod +w runtime/system.lsp
\tcp -p sys/unix/~A/system.lsp runtime/system.lsp
\tchmod -w runtime/system.lsp

" system-name system-name)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    (dependencies outf system-name)
    (format outf (system-rules))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    (format outf
        "misc/intgen: misc/intgen.c
\tcd misc; make intgen

misc/unpacker: misc/unpacker.c misc/convert.c
\tcd misc; make unpacker

misc/packer: misc/packer.c misc/convert.c
\tcd misc; make packer

nyqsrc/sndfnintptrs.h: $(NYQHDRS) snd/snd.h misc/intgen
\t$(INTGEN) nyqsrc/sndfnint $(NYQHDRS)

nyqsrc/seqfnintptrs.h: $(CMTHDRS) misc/intgen
\t$(INTGEN) nyqsrc/seqfnint $(CMTHDRS)

clean:
\tcd misc; make clean
\trm -f $(OBJECTS)
# These could be deleted, but they're part of the release, so we won't
# Note that these files are machine-generated:
# \trm -f nyqsrc/sndfnintptrs.h nyqsrc/sndfnint.c nyqsrc/sndfnintdefs.h
# \trm -f nyqsrc/seqfnintptrs.h nyqsrc/seqfnint.c nyqsrc/seqfnintdefs.h

cleaner: clean
\tcd misc; make cleaner
\trm -f *.backup */*.backup
\trm -f *~~ */*.*~~
\trm -f #*# */#*#
\trm -f *.save */*.save
\trm -f *.CKP */*.CKP
\trm -f *.BAK */*.BAK
\trm -f *.old */*.old
\trm -f *.gold */*.gold
\trm -f playparms
\trm -f points.dat
\trm -f core
\trm -f $(NY)

release: cleaner
\tcd misc; make packer
\tmisc/packer files.txt release.pac
\trm -f *.wav
\tmv ny ..
\tmv -f *.pac ..
\trm -f unpacker
\trm -f packer
\tcd ..; zip -r nyquist.zip nyquist
\t../ny misc/cmu/cmu-linux-install.lsp
\tmv ../ny ./ny
")

    (cond ((eq system-type 'rs6k)
       (format outf "
tar: cleaner
\tsh -v sys/unix/cmu/tar.script

backup: cleaner
\tsh -v sys/unix/cmu/backup.script
")))
    (close outf)
    ))

;; system-defs looks for a string of system-dependent defs for the makefile
;;
(defun system-defs () (system-var "-DEFS"))


;; system-rules looks for a string of system-dependent rules for the makefile
;;
(defun system-rules () (system-var "-RULES"))


;; system-var returns a string stored in the variable (if any):
;;   <system-type>-<suffix>
;;
(defun system-var (suffix)
  (let ((v (intern (strcat (symbol-name system-type) suffix))))
    (cond ((boundp v) (symbol-value v))
      (t ""))))


;; object-files - writes names of all object files for linking
;;
(defun object-files (outf)
  (let ((flist (append (add-prefix "xlisp/" xlfiles)
               (add-prefix "tran/" transfiles)
               (add-prefix "cmt/" cmtfiles)
               (add-prefix "nyqsrc/" nyqfiles)
               (add-prefix "nyqstk/src/" stksrcfiles)
               (add-prefix "nyqstk/" stkfiles)
               (add-prefix "fft/" fftfiles)
               (add-prefix "nyqsrc/" intfiles)
               (add-prefix "snd/" sndfiles)
               (add-prefix "portaudio/" pafiles)
               '("sys/unix/osstuff" "sys/unix/term"))))
    (setf flist (add-suffix flist ".o"))
    (format outf "OBJECTS = ")
    (write-file-list outf flist #\\)))


;; add-prefix - place string at beginning of each string in list
;;
(defun add-prefix (prefix lis)
  (mapcar #'(lambda (str) (strcat prefix str)) lis))


;; add-suffix - place string at end of each string in list
;;
(defun add-suffix (lis suffix)
  (mapcar #'(lambda (str) (strcat str suffix)) lis))


;; write-file-list - write file names to Make macro
;;
(defun write-file-list (outf flist continuation-char)
  (while flist
     (dotimes (i 4)
          (format outf "~A " (car flist))
          (setf flist (cdr flist))
          (if (null flist) (return)))
     (if flist (format outf " ~A~%\t" continuation-char)))
  (format outf "~%~%"))


(defun nyqheaders (outf)
  (let ((flist (append
                (list "snd/snd")
                 (add-prefix "nyqsrc/" nyqsrcfiles)
                 (add-prefix "tran/" transfiles))))
    (setf flist (mapcar #'(lambda (f) 
                            ;; exception: sndwritepa.h -> sndwrite.h
                            (cond ((equal f "nyqsrc/sndwritepa")
                                   "nyqsrc/sndwrite.h")
                                  (t
                                   (strcat f ".h"))))
                        flist))
    (format outf "NYQHDRS = ")
    (write-file-list outf flist #\\)))


(defun cmtheaders (outf)
  (let ((flist 
     (append '("cmt/seqdecls" "nyqsrc/seqext" "cmt/seq"
           "nyqsrc/seqinterf")  ; order is important here!
         (add-prefix "cmt/"
          '("seqread" "seqmread" "seqwrite" "seqmwrite")))))
    (setf flist (add-suffix flist ".h"))
    (format outf "CMTHDRS = ")
    (write-file-list outf flist #\\)))


(defun dependencies (outf system-name)
  ;; this forces generation of sndfnintdefs.h, seqfnintdefs.h:
  (dolist (f (append (add-prefix "nyqsrc/" nyqfiles)
             (add-prefix "snd/" sndfiles)
             (add-prefix "fft/" fftfiles)
             (add-prefix "tran/" transfiles)
             (add-prefix "nyqsrc/" intfiles)))
    (let ((ex (assoc f depends-exceptions :test #'equal)))
      (cond ((and ex (cdr ex))
         (format outf "~A.o: ~A.c ~A~%" f f (cadr ex))
         (format outf "\t$(CC) -c ~A.c -o ~A.o $(CFLAGS)~%~%" f f))
        (t
         (format outf "~A.o: ~A.c ~A.h nyqsrc/sound.h nyqsrc/falloc.h nyqsrc/cque.h~%"
              f f f)
         (format outf "\t$(CC) -c ~A.c -o ~A.o $(CFLAGS)~%~%" f f)))))
  (dolist (f stkfiles)
     (format outf "nyqstk/~A.o: nyqstk/~A.cpp nyqstk/~A.h~%"
             f f f)
     (format outf "\tg++ -c nyqstk/~A.cpp -o nyqstk/~A.o $(CFLAGS)~%~%"
             f f))

  (dolist (f stksrcfiles)
     (format outf "nyqstk/src/~A.o: nyqstk/src/~A.cpp nyqstk/include/~A.h~%"
             f f f)
     (format outf "\tg++ -c nyqstk/src/~A.cpp -o nyqstk/src/~A.o $(CFLAGS)~%~%"
             f f))

  (format outf "xlisp/xlftab.o: nyqsrc/sndfnintptrs.h nyqsrc/sndfnintdefs.h")
  (format outf " nyqsrc/seqfnintptrs.h nyqsrc/seqfnintdefs.h~%")
  (format outf "\t$(CC) -c xlisp/xlftab.c -o xlisp/xlftab.o $(CFLAGS)~%~%")
  (dolist (f (append (add-prefix "xlisp/" xlfiles)
             (add-prefix "cmt/" cmtfiles)
             '("sys/unix/osstuff")))
    (cond ((and (not (equal f "xlisp/xlftab"))   ; special case handled above
        (not (and (equal f "xlisp/xljump") ; case handled below
              (equal system-name "next"))))
       (format outf "~A.o: ~A.c~%\t$(CC) -c ~A.c -o ~A.o $(CFLAGS)~%~%"
           f f f f)))))


;; MAKESRC -- makes the Makesrc file
;;
(defun makesrc ()
  (let (outf)
    (load "misc/transfiles.lsp") ; just to make sure we're current
    (format t "Writing Makesrc\n")
    (setf outf (open "Makesrc" :direction :output))
    (format outf "#
# FILE: Makefile to import sources
#
# usage: make -f Makesrc
#
# I want to be able to keep sources separate from objects, e.g. the
# MIDI sequence code comes from the CMU MIDI Toolkit, and XLisp should
# be compile-able without Nyquist.
#
# There are tools to do this, but I don't expect to find the same tool
# everywhere Nyquist is ported, so I'm taking this simple path.

# Source Paths:
NYQ = nyqsrc
CMT = cmt
XLISP = xlisp
TRANS = tran
SND = snd

EVERYTHING: sources

")
    (source-files outf)
    (format outf "
#
# XLISP SOURCES
#
")
    (source-depends outf xlfiles ".c" "XLISP")
    (source-depends outf xlfiles-h ".h" "XLISP")
    (source-depends outf xlfiles-lsp ".lsp" "XLISP")

    (format outf "
#
# NYQUIST SOURCES
#
")
    (source-depends outf nyqfiles ".c" "NYQ")
    (source-depends outf (exceptions-filter nyqfiles) ".h" "NYQ")
    (source-depends outf nyqfiles-h ".h" "NYQ")
    (source-depends outf nyqfiles-lsp ".lsp" "NYQ")
    (source-depends outf extrafiles "" "NYQ")

    (format outf "
#
# CMT SOURCES
#
")
    (source-depends outf cmtfiles ".c" "CMT")
    (source-depends outf (exceptions-filter cmtfiles) ".h" "CMT")
    (source-depends outf cmtfiles-h ".h" "CMT")

    (format outf "
#
# TRANS SOURCES
#
")
    (source-depends outf transfiles ".c" "TRANS")
    (source-depends outf transfiles ".h" "TRANS")

    (format outf "
#
# SND SOURCES
#
")
    (source-depends outf sndfiles ".c" "SND")
    (source-depends outf sndfiles ".h" "SND")

    (close outf)))
        

;; SOURCE-FILES -- writes the sources: list to Makesrc
;;
(defun source-files (outf)
  (setf flist (append xlfiles nyqfiles fftfiles cmtfiles transfiles sndfiles))
  (setf flist-h (append xlfiles-h nyqfiles-h cmtfiles-h transfiles))
  ; Add .c files as .h files unless exceptions list indicates there is
  ; no .h file:
  (setf flist-h (append (exceptions-filter nyqfiles)
            (exceptions-filter cmtfiles)
            (exceptions-filter sndfiles)
            flist-h))
  (setf flist-lsp (append xlfiles-lsp nyqfiles-lsp sndfiles-lsp))
  (setf flist (mapcar #'(lambda (f) (strcat f ".c")) flist))
  ;; handle c++ files from stk:
  (setf stk-cpp (mapcar #'(lambda (f) (strcat f ".cpp")) stkfiles))
  (setf flist (append flist stk-cpp))
  (setf flist-h (mapcar #'(lambda (f) (strcat f ".h")) flist-h))
  (setf flist-lsp
    (mapcar #'(lambda (f) (strcat f ".lsp")) flist-lsp))
  (format outf "sources: ")
  (write-file-list outf (append flist flist-h flist-lsp extrafiles) #\\))


;; EXCEPTIONS-FILTER - remove .h files from list
; the depends-exceptions tells whether a .h file exists for a .c file
;; 
(defun exceptions-filter (files)
  (let (result)
    (dolist (f files)
      (let ((ex (assoc f depends-exceptions :test #'equal)))
    (cond (ex
           (if (and (cdr ex)
              (string-search (strcat f ".h") (cadr ex)))
           (push f result)))
          (t (push f result)))))
    result))


;; SOURCE-DEPENDS -- write dependency for source files
;;
(defun source-depends (outf files ext dir)
  (dolist (f files)
    (let ((fname (strcat f ext)))
      (format outf
 "~A: $(~A)/~A~%\trm -f ~A~%\tcp -p $(~A)/~A .~%\tchmod -w ~A~%~%"
          fname dir fname fname dir fname fname))))


;;===================================================
;; SYSTEM DEPENDENCIES
;;===================================================

(setf rs6k-defs "
MIDI = /afs/cs/project/music/rs6k/midilib
CC = cc
# change -g to -O for optimization
# to disable command line editing, take out -DREADLINE
CFLAGS = -DREADLINE -DCMTSTUFF -g $(INCL) -I$(MIDI)
XFLAGS = $(CFLAGS) -qlanglvl=extended
LN = xlc -qattr -qlist
# to disable command line editing, take out -lreadline -lcurses
LFLAGS = -lm -lreadline -lcurses -lpthread -L$(MIDI) -lmidi -lbsd -lg
")


(setf next-defs "
CC = cc
# to disable command line editing, take out -DREADLINE
CFLAGS = -DREADLINE -DCMTSTUFF -O $(INCL)
LN = cc
# to disable command line editing, take out -lreadline -lcurses
LFLAGS = -lm -lreadline -lcurses -lpthread
")

(setf next-rules "
# this doesn't compile with the -O switch (a NeXT compiler bug?)
xlisp/xljump.o : xlisp/xljump.c xlisp/xlisp.h
\t$(CC) -DCMTSTUFF -c xlisp/xljump.c -o xlisp/xljump.o
")

(setf pmax-defs "
CC = cc
# to disable command line editing, take out -DREADLINE
CFLAGS = -DREADLINE -DCMTSTUFF -g $(INCL)
LN = cc
# to disable command line editing, take out -lreadline -lcurses
LFLAGS = -lm -lreadline -lcurses
")


(setf sgi-defs "
CC = cc
# to disable command line editing, take out -DREADLINE
CFLAGS = -DREADLINE -DCMTSTUFF -g $(INCL)
LN = cc
# to disable command line editing, take out -lreadline -lcurses
LFLAGS = -lm -lreadline -lcurses -lpthread
# you would need -lmd if UNIX_IRIX_MIDIFNS were defined in midifns.c
")


(setf sparc-defs "
CC = gcc
# to disable command line editing, take out -DREADLINE
CFLAGS = -DREADLINE -DCMTSTUFF -g $(INCL)
LN = g++
# to disable command line editing, take out -lreadline -lcurses
LFLAGS = -lm -lreadline -lcurses -lpthread
")

(setf linux-defs "
CC = gcc
# to enable command line editing, use -DREADLINE. WARNING: THIS WILL DISABLE THE
# ABILITY TO INTERRUPT LISP AND USE SOME OTHER HANDY CONTROL CHARACTERS
# using OSS because at the time of testing, ALSA had problems unless the number of channels
#   matched the hardware
CFLAGS = -DCMTSTUFF -DPA_LITTLE_ENDIAN -g $(INCL) -DHAVE_LIBPTHREAD=1 -DPA_USE_OSS=1
LN = g++
# to disable command line editing, take out -lreadline -lcurses
LFLAGS = -lm -lreadline -lcurses -lpthread -lasound
TAGS:
	find . \( -name "*.c" -o -name "*.h" \) -print | etags -

tags: TAGS
")







