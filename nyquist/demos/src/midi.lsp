(defun seq-midi-sal (seq note &optional ctrl bend touch prgm)
  (seq-midi seq (note (chan pitch vel) (funcall note chan pitch vel))
    (ctrl (chan num val) (if ctrl (funcall ctrl chan num val)))
    (bend (chan val) (if bend (funcall bend chan val)))
    (touch (chan val) (if touch (funcall touch chan val)))
    (prgm (chan val) (if prgm (funcall prgm chan val)))))
