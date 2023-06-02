from setuptools import setup

setup(
    name='vaetfg',
    version='0.1',
    packages=['utils', 'utils.losses', 'utils.losses.images', 'utils.losses.images.application', 'utils.batches',
              'utils.batches.domain', 'utils.batches.domain.exceptions', 'utils.batches.application',
              'utils.batches.infrastructure', 'utils.epsilons', 'utils.epsilons.domain', 'utils.epsilons.application',
              'utils.epsilons.infrastructure', 'utils.external', 'utils.external.fid', 'utils.external.tvd', 'project',
              'project.domain', 'project.domain.exceptions', 'project.infrastructure', 'project.infrastructure.images',
              'project.infrastructure.images.main'],
    package_dir={'': 'src'},
    url='https://github.com/ikerRAD/VAE-TFG.git',
    license='MIT',
    author='IkerPG',
    author_email='ikerpingar@gmail.com',
    description='Bachellor\'s thesis to research about the VAE'
)
