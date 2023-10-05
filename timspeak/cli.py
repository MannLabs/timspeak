#!python


# external
import click

# local
import timspeak


@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)
@click.pass_context
@click.version_option(timspeak.__version__, "-v", "--version")
def run(ctx, **kwargs):
    name = f"timspeak { timspeak.__version__}"
    click.echo("*" * (len(name) + 4))
    click.echo(f"* {name} *")
    click.echo("*" * (len(name) + 4))
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("gui", help="Start graphical user interface.")
def gui():
    import timspeak.gui
    timspeak.gui.run()


@run.command("run_pipeline", help="Run timspeak execution_pipeline.", no_args_is_help=True)
@click.argument("configfile", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None, required=True)
@click.argument("samplefile", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False)
@click.argument("outputfile", type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None, required=False)
def run_pipeline(
    configfile: str,
    samplefile: str = None,
    outputfile: str = None,
):
    import os
    configfile = os.path.abspath(configfile)
    if samplefile is not None:
        samplefile = os.path.abspath(samplefile)
    if outputfile is not None:
        outputfile = os.path.abspath(outputfile)
    import timspeak.main
    timspeak.main.main(configfile, samplefile, outputfile)


if __name__ == "__main__":
    run()
