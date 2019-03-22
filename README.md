# Siamese Capsule Networks

This repository contains the code to implement  *Siamese Capsule Networks* for
face verification.

You will need to follow the following steps:

* First run python setup.py install. If you have a GPU this should install
  fine on linux but you may have to install directly on pytorch.org and choose
  the correct settings for your cuda driver version.

* Then download "All image with funnelling" from here http://vis-www.cs.umass.edu/lfw/#deepfunnel-anchor
  and place data/lfw/. Similarly download the AT\&T dataset training and test
  splits from here - https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
  and store in data/att_faces

* Make sure the names of the folder paths
