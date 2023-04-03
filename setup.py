from setuptools import setup


_config = {
    "name": "pyturbo-aero",
    "version":"1.0.0",
    "url": "https://gitlab.grc.nasa.gov/lte-turbo/pyturbo",
    "author": "Paht Juangphanich",
    "author_email": "paht.juangphanich@nasa.gov",
    "packages":["pyturbo","pyturbo.helper","pyturbo.aero"],
    "install_requires":['plotly','tqdm','scipy','pandas','numpy','matplotlib','numpy-stl'],
    'license':"GNU GPLv3"
}

def main() -> int:
    """ Execute the setup command.
    """

    def version():
        """ Get the local package version. """
        return _config["version"]

    _config.update({
        "version": version(),
    })

    setup(**_config)
    return 0


# Make the script executable.
if __name__ == "__main__":
    raise SystemExit(main())
