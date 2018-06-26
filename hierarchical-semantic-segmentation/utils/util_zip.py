import zipfile
import os


def zipit(path, archname):
  """
  Inspired from https://gist.github.com/felixSchl/d38b455df8bf83a78d3d
  :param path: directory that you want to zip
  :param archname: full path and name of the archive
  :return:
  """
  print(path)
  # if os.path.isdir(path):
  #   assert not os.path.exists(path)
  #   os.makedirs(path)
  if not os.path.exists(os.path.split(archname)[0]):
    os.makedirs(os.path.split(archname)[0])
  archive = zipfile.ZipFile(archname, "w", zipfile.ZIP_DEFLATED)
  if os.path.isdir(path):
    _zippy(path, path, archive)
  else:
    _, name = os.path.split(path)
    archive.write(path, name)
  archive.close()


def _zippy(base_path, path, archive):
  paths = os.listdir(path)
  for p in paths:
    p = os.path.join(path, p)
    if os.path.isdir(p):
      _zippy(base_path, p, archive)
    else:
      if os.path.splitext(p)[1] == '.py':
        archive.write(p, os.path.relpath(p, base_path))


if __name__ == '__main__':
  # Example use
  zipit('/home/mps/Documents/semantic-segmentation-fork/semantic-segmentation', 'test.zip')
