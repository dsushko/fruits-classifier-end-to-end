import click
from pipeline import ModelRunner


@click.command('fruits-classifier')
@click.argument('build', type=click.Choice(['baseline']))
@click.option(
    '--validate', '-v',
    help='Defines whether to make validation step',
    is_flag=True
)
@click.option(
    '--data-folder', '-f',
    help='Defines whether to make validation step',
    type=click.Path(exists=True), default='./data/', show_default=True
)
@click.option(
    '--cfg-path', '-cfg',
    help='Defines folder where modeling configs are stored',
    type=click.Path(exists=True), default='cfg/model/', show_default=True
)
def main(build: str = 'baseline',
         validate: bool = False,
         data_folder: str = './data/',
         cfg_path: str = './cfg/model/'):
    cfg_path = cfg_path + build + '.yaml'
    running_pipeline = ModelRunner(cfg_path, data_folder, validate)
    running_pipeline.run()
    

if __name__ == '__main__':
    main()
