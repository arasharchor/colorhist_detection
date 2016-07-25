#!/usr/bin/env python
"""
Module defining the LINE-MOD detector to find objects in a scene
"""

from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward
from object_recognition_core.db import Document, Documents
from ecto_opencv import calib, features2d, highgui
from ecto_opencv.features2d import FeatureDescriptor
from object_recognition_core.pipelines.detection import DetectorBase
from object_recognition_colorhist import colorhist_detection
import ecto
from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward

########################################################################################################################

class ColorHistDetector(ecto.BlackBox, DetectorBase):
    def __init__(self, *args, **kwargs):
        print "test0"
        ecto.BlackBox.__init__(self, *args, **kwargs)
        DetectorBase.__init__(self)

    @staticmethod
    def declare_cells(p):
        print "test 1_1"

        # passthrough cells
        cells = {'json_db': CellInfo(ecto.Constant)}
                 #'object_id': CellInfo(ecto.Constant),
                
        cells.update({'model_reader': colorhist_detection.ModelReader(),
                      'detector': CellInfo(colorhist_detection.Detector)})        
        print "test1_2"

        return cells

    @classmethod
    def declare_forwards(cls, _p):
        print "test2"
        p = {'json_db': [Forward('value', 'json_db')]}
             #'object_id': [Forward('value', 'object_id')]}
        p.update({'detector': 'all'})
        i = {}
        o = {'detector': [Forward('pose_results')]}

        return (p,i,o)

    @classmethod
    def declare_direct_params(self, p):
        print "test3_1"
        p.declare('json_object_ids', 'The ids of the objects to find as a JSON list or the keyword "all".', 'all')
        print "test3_2"

    def connections(self, p):
        print "test4"
        connections = [ self.json_db[:] >> self.model_reader['json_db'] ]
       # self.object_id[:] >> self.model_reader['object_id'] ]
        connections += [ self.model_reader['colorValues'] >> self.detector['colorValues'] ]
        print "test5"
        
        return connections
