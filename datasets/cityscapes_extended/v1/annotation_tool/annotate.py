
import glob, sys, inspect, os, argparse

import numpy as np
import PIL.Image as Image
from scipy import misc, ndimage
from skimage import morphology

from PyQt5.QtCore import QSize, Qt #, pyqtSlot
from PyQt5.QtGui import QImage, QIcon, QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QGridLayout, QVBoxLayout, QSizePolicy,
                             QHBoxLayout, QPushButton, QStyleOptionButton, QStyle, QStatusBar)

# TODO: shrink candidate traffic sign image if larger than window height/width
# TODO: smooth exit when no more packets exist: now an iteration stop is thrown
#         at line 103, in update: self.packet = next(self.loader.generator)


#for k,v in vars(QtGui.QImage).items():
  #print(k, '\t\t', v, '\n')

#for m in dir(QtGui.QImage):
  #print(m)

#for m in inspect.getmembers(QtGui.QImage): #, predicate=inspect.ismethod)
  #print(m)
  
#print(inspect.signature(QGridLayout.addWidget))


class Packet(object):
  def __init__(self, image_and_label_fpath):
    self.ipath = image_and_label_fpath[0]
    self.lpath = image_and_label_fpath[1]
    #self.ldir, self.lname, self.lformat = Loader.split_path(self.lpath)
    self.image = misc.imread(self.ipath)
    self.label = misc.imread(self.lpath)
    # one run to remove small signs
    temp_signs_mask = self.label==20
    temp_labeled, _ = ndimage.measurements.label(temp_signs_mask)
    # temp_locs[0] --> component 1, temp_locs[1] --> component 2, ...
    temp_locs = ndimage.find_objects(temp_labeled)
    ignore_component_ids = []
    for i, loc in enumerate(temp_locs):
      width = loc[1].stop - loc[1].start
      height = loc[0].stop - loc[0].start
      if width<20 or height<20:
        ignore_component_ids.append(i+1)
    self.signs_mask = temp_signs_mask
    for i in ignore_component_ids:
      self.signs_mask -= temp_labeled==i
    # second run
    #structure = [[0,0,1,0,0],
                 #[0,1,1,1,0],
                 #[1,1,1,1,1],
                 #[0,1,1,1,0],
                 #[0,0,1,0,0]]
    #structure = morphology.disk(7, dtype=np.int32)
    # the biggest structure that doesn't give error...
    structure = [[1,1,1],
                 [1,1,1],
                 [1,1,1]]
    self.labeled, self.Nsigns = ndimage.measurements.label(self.signs_mask, structure=structure)
    self.locs = ndimage.find_objects(self.labeled)
    self.signs = [self.image[loc] for loc in self.locs]
    self.choices = []


class Loader(object):
  @staticmethod
  def split_path(path):
    # filepath = head/root.ext[1:]
    head, tail = os.path.split(path)
    root, ext = os.path.splitext(tail)
    return head, root, ext[1:]

  def __init__(self, image_and_label_fpaths):
    # ilfpaths: iterator (zip object)
    self.ilfpaths = list(image_and_label_fpaths)
    self.counter = 0
    self.Npaths = len(list(self.ilfpaths))
    #print('debug:N', self.Npaths)
    # list of read pairs of images and labels
    #self.read_packets = []
    self.generator = self._next_packet()

  def _next_packet(self):
    for ilfp in self.ilfpaths:
      self.counter += 1
      packet = Packet(ilfp)
      yield packet


class CurrentPacket(object):  
  def __init__(self, loader):
    self.loader = loader
    self.packet = None
  
  def update(self, choice):
    self.packet.choices.append(choice)
    # reached last sign of the packet
    if len(self.packet.choices)==self.packet.Nsigns:
      if self.packet.Nsigns:
        save_annotations(self.packet)
      # load next packet till usefull signs are found
      while True:
        self.packet = next(self.loader.generator)
        if self.packet.Nsigns:
          break
        else:
          save_no_annotations(self.packet)
      #print('debug:new packet:', self.packet.ipath)
    return self.packet.signs[len(self.packet.choices)]


def save_annotations(packet):
  # label: HxW, ground truth of standard cityscapes
  # traffic signs: [0,42]: known, 43: unknown/multiple, 44: not traffic sign
  # labeled: HxW, connected components of traffic sign mask,
  #   output of ndimage.measurements.label, connected regions are labeled from 1
  # choices: list of choices corresponding to labeled
  #   ex. for 5 signs: [4,0,43,6,44]: that is component 1 is traffic sign with id 4, ...
  
  # add mapping for zero feature and add offset of 2000
  
  #print('debug:choices:', packet.choices)
  choices = np.array([0] + list(map(lambda x: x+2000, packet.choices)))
  #print('debug:choices:', choices)
  # component 0 will be painted with choices[0]=2000, ...
  mapped_labeled = choices[packet.labeled]
  # make 2043 and 2044 pixels 0 so original label is used
  mask = np.logical_or(mapped_labeled==2043, mapped_labeled==2044)
  mapped_labeled -= mapped_labeled*mask
  
  new_label = (packet.label*(mapped_labeled==0) + mapped_labeled).astype(np.int32)
  
  new_path = packet.lpath.replace('cityscapes', 'cityscapes_extended')
  #print('debug:new_path:', new_path)
  #assert not os.path.exists(new_path)
  # imsave cannot create directories recursively
  if not os.path.exists(split_path(new_path)[0]):
    os.makedirs(split_path(new_path)[0])
  if os.path.exists(new_path):
    print(f"WARNING: '{new_path}' already exists, delete manually this image and restart if you want to create new annotation.")
  else:
    Image.fromarray(new_label, mode='I').save(new_path) # 32-bit signed integer 


def save_no_annotations(packet):
  # when there are no signs usefull signs
  new_label = packet.label.astype(np.int32)
  new_path = packet.lpath.replace('cityscapes', 'cityscapes_extended')
  if not os.path.exists(split_path(new_path)[0]):
    os.makedirs(split_path(new_path)[0])
  if os.path.exists(new_path):
    print(f"WARNING: '{new_path}' already exists, delete manually this image and restart if you want to create new annotation.")
  else:
    Image.fromarray(new_label, mode='I').save(new_path) # 32-bit signed integer 


def get_image_and_label_fpaths(cityscapes_path):
  # paths for train, val, val/delete_all
  pt = cityscapes_path + '/leftImg8bit/train/*/*.png'
  pv = cityscapes_path + '/leftImg8bit/val/*/*.png'
  pvd = cityscapes_path + '/leftImg8bit/val/delete_all/*.png'
  # remove already annotated images
  annotated = glob.glob(cityscapes_path.replace('cityscapes', 'cityscapes_extended') + '/gtFine/*/*/*.png')
  annotated = [a.replace('cityscapes_extended', 'cityscapes')
                .replace('gtFine/', 'leftImg8bit/')
                .replace('gtFine_labelIds.png', 'leftImg8bit.png') for a in annotated]
  image_fnames = sorted(list(set(glob.glob(pt) + glob.glob(pv)) - set(glob.glob(pvd)) - set(annotated)))
  print('debug:image_fnames:', len(image_fnames))
  label_fnames = [ef.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                    .replace('leftImg8bit/', 'gtFine/')  for ef in image_fnames]
  return zip(image_fnames, label_fnames)


def split_path(path):
  # filepath = head/root.ext[1:]
  head, tail = os.path.split(path)
  root, ext = os.path.splitext(tail)
  return head, root, ext[1:]


class myPushButton(QPushButton):
  # extend QPushButton for strechable QIcon
  def __init__(self, label=None, parent=None):
    super().__init__(label, parent)
    self.pad = 2     # padding between the icon and the button frame
    self.minSize = 4 # minimum size of the icon
    sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.setSizePolicy(sizePolicy)
  def paintEvent(self, event):
    qp = QPainter()
    qp.begin(self)
    # get default style
    opt = QStyleOptionButton()
    self.initStyleOption(opt)
    # scale icon to button size
    Rect = opt.rect
    h = Rect.height()
    w = Rect.width()
    iconSize = max(min(h, w) - 2 * self.pad, self.minSize)
    opt.iconSize = QSize(iconSize, iconSize)
    # draw button
    self.style().drawControl(QStyle.CE_PushButton, opt, qp, self)
    qp.end()
        

class App(QWidget):
  def __init__(self, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('cityscapes_path',
                        type=str,
                        help='Cityscapes path without / at the end (train and val ' +
                             'leftImg8bit folders must be the originals (unchanged).')
    args = parser.parse_args(argv)
    print('Results are saved using the same folder and name structure as Cityscapes' +
          ' in a new cityscapes_extended folder under parent of cityscapes folder.')
    
    super().__init__()
    
    self.imageLabel = None
    self.signLabel = None
    self.loader = Loader(get_image_and_label_fpaths(args.cityscapes_path))
    self.current_packet = CurrentPacket(self.loader)
    self.initUI()
    
  def initUI(self):
    ## create widgets
    # traffic signs
    self.signButtons = []
    for i, p in enumerate(sorted(glob.glob('resources/*'))):
      sign = myPushButton()
      sign.setIcon(QIcon(QPixmap(QImage(p).scaled(60, 60))))
      #sign.resize(64, 64)
      sign.setMaximumSize(68, 68)
      sign.clicked.connect(self.signButtonClicked_fn(i, self.updateUI))
      self.signButtons.append(sign)
    # image
    self.imageLabel = QLabel()
    # sign
    self.signLabel = QLabel()
    # status bar
    self.messageLabel = QLabel('Panagiotis Meletis, PhD student, TUE, Netherlands. May, 2017.')
    self.progressLabel = QLabel()
    self.statusBar = QStatusBar()
    self.statusBar.addWidget(self.messageLabel)
    self.statusBar.addPermanentWidget(self.progressLabel)
    
    # initiate
    # load next packet till usefull signs are found
    while True:
      self.current_packet.packet = next(self.current_packet.loader.generator)
      if self.current_packet.packet.Nsigns:
        break
      else:
        save_no_annotations(self.current_packet.packet)
    next_sign = self.current_packet.packet.signs[len(self.current_packet.packet.choices)]
    
    self.updateUI(next_sign)

    ## create layouts
    # sign Buttons layout
    signButtonsLayout = QGridLayout()
    for i, sb in enumerate(self.signButtons):
      # by default QPushButton has vertical policy Fixed
      #temp_policy.setHorizontalPolicy(QSizePolicy.Preferred)
      #temp_policy.setVerticalPolicy(temp_policy.horizontalPolicy())
      #print(temp_policy.horizontalPolicy(), temp_policy.verticalPolicy())
      #sb.setSizePolicy(temp_policy)
      signButtonsLayout.addWidget(sb, i//16, i%16)
    # image and sign layout
    imageSignLayout = QHBoxLayout()
    imageSignLayout.addWidget(self.imageLabel)
    imageSignLayout.addStretch()
    imageSignLayout.addWidget(self.signLabel)
    imageSignLayout.addStretch()
    # statusBar layout
    statusBarLayout = QVBoxLayout()
    statusBarLayout.addStretch()
    statusBarLayout.addWidget(self.statusBar)
    # main layout
    layout = QVBoxLayout()
    layout.addLayout(imageSignLayout)
    layout.addLayout(signButtonsLayout)
    layout.addLayout(statusBarLayout)
    self.setLayout(layout)
    self.setWindowTitle('Cityscapes (train, val) Traffic Sign Annotator using' +
                        ' the German Traffic Sign Detection Benchmark signs')
    #self.setGeometry(self.left, self.top, self.width, self.height)
    self.setWindowState(Qt.WindowMaximized)
    self.show()
  
  def updateUI(self, next_sign):
    ## TODO: setPixmap only if image is updated
    self.imageLabel.setPixmap(QPixmap(QImage(self.current_packet.packet.ipath).scaled(1024, 512)))
    # enlarge small sign images
    if next_sign.shape[0]<50 or next_sign.shape[1]<50:
      next_sign = misc.imresize(next_sign, 3.0, interp='lanczos')
    bytesPerLine = next_sign.strides[1]*next_sign.shape[1] #bytesPerComponent * width
    self.signLabel.setPixmap(QPixmap(QImage(next_sign.tostring(),
                                            next_sign.shape[1],
                                            next_sign.shape[0],
                                            bytesPerLine,
                                            QImage.Format_RGB888)))
    
    self.progressLabel.setText(f"{self.current_packet.loader.counter} / {self.current_packet.loader.Npaths} files annotated")
    
  def signButtonClicked_fn(self, signId, updateUI):
    assert callable(updateUI)
    def signButtonClicked():
      next_sign = self.current_packet.update(signId)
      updateUI(next_sign)
    return signButtonClicked


if __name__ == '__main__':
  app = QApplication(sys.argv)
  ex = App(sys.argv[1:])
  sys.exit(app.exec_())


