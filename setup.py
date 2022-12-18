from setuptools import setup

setup(name='ubt',
      version='0.1',
      description='tracking framework with UpperBoundTracker',
      url='http://github.com/dolokov/upper_bound_tracking',
      author='Alexander Dolokov',
      author_email='alexander.dolokov@gmail.com',
      license='MIT',
      packages=['src/ubt'],
      zip_safe=False)


## sudo apt-get install libsqlite3-dev