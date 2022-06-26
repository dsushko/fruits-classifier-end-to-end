import os

import click
from pipeline import ModelRunner
from utils.globalparams import GlobalParams

@click.command('fruits-classifier')
@click.argument('build', type=click.Choice(['baseline', 'vgg16']))
@click.option(
    '--validate-flag', '-v',
    help='Defines whether to make validation step',
    is_flag=True
)
def main(build: str = 'baseline',
         validate_flag: bool = False):

    GlobalParams().update_params({'build': build})
    running_pipeline = ModelRunner(build, validate_flag)
    running_pipeline.run()
    

if __name__ == '__main__':
    main()
