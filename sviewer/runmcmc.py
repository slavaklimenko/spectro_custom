from adjustText import adjust_text
import astroplan
import astropy.coordinates
from astropy.cosmology import Planck15
from astropy.io import ascii, fits
from astropy.table import Table
import astropy.time
from astroquery import sdss as aqsdss
from chainconsumer import ChainConsumer
from collections import OrderedDict
from copy import deepcopy, copy
import emcee
import h5py
from importlib import reload
import julia
from lmfit import Minimizer, Parameters, report_fit, fit_report, conf_interval, printfuncs, Model
from matplotlib.colors import to_hex
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from multiprocessing import Process
import numdifftools as nd
import pandas as pd
import pickle
import os
import platform
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QWidget,
                             QFileDialog, QTextEdit, QVBoxLayout,
                             QSplitter, QFrame, QLineEdit, QLabel, QPushButton, QCheckBox,
                             QGridLayout, QTabWidget, QFormLayout, QHBoxLayout, QRadioButton,
                             QTreeWidget, QComboBox, QTreeWidgetItem, QAbstractItemView,
                             QStatusBar, QMenu, QButtonGroup, QMessageBox, QToolButton, QColorDialog)
from PyQt5.QtCore import Qt, QPoint, QRectF, QEvent, QUrl, QTimer, pyqtSignal, QObject, QPropertyAnimation
from PyQt5.QtGui import QDesktopServices, QPainter, QFont, QColor, QIcon
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.special import erf
from scipy.stats import gaussian_kde
import sfdmap
from shutil import copyfile
import subprocess
import tarfile
from threading import Thread
from ..a_unc import a
from ..absorption_systems import vel_offset
from ..atomic import *
from ..plot_spec import *
from ..profiles import add_LyaForest, add_ext, add_ext_bump, add_LyaCutoff, convolveflux, tau
from ..stats import distr1d, distr2d
from ..XQ100 import load_QSO
from .console import *
from .external import spectres
from .erosita import *
from .fit_model import *
from .fit import *
from .graphics import *
from .lines import *
from .sdss_fit import *
from .tables import *
from .obs_tool import *
from .colorcolor import *
from .utils import *

import sviewer as sv

if __name__ == '__main__':

    filename = ''
    s = sv.sviewer()
    s.openFile(filename=filename)