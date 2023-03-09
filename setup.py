#setup.py for ruth-nlu
import pathlib

import setuptools


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')
"""
use following requiresments 
python = ">=3.6,<3.9"
boto3 = "^1.12"
requests = "^2.23"
matplotlib = ">=3.1,<3.4"
attrs = ">=19.3,<21.3"
jsonpickle = ">=1.3,<2.1"
redis = "^3.4"
numpy = ">=1.16,<1.19"
scipy = "^1.4.1"
absl-py = ">=0.9,<0.14"
apscheduler = ">=3.6,<3.8"
tqdm = "^4.31"
networkx = ">=2.4,<2.6"
fbmessenger = "~6.0.0"
pykwalify = ">=1.7,<1.9"
coloredlogs = ">=10,<16"
"ruamel.yaml" = "^0.16.5"
scikit-learn = ">=0.22,<0.25"
slackclient = "^2.0.0"
twilio = ">=6.26,<6.51"
webexteamssdk = ">=1.1.1,<1.7.0"
mattermostwrapper = "~2.2"
rocketchat_API = ">=0.6.31,<1.17.0"
colorhash = "~1.0.2"
jsonschema = "~3.2"
packaging = ">=20.0,<21.0"
pytz = ">=2019.1,<2022.0"
rasa-sdk = "^2.8.0"
colorclass = "~2.2"
terminaltables = "~3.1.0"
sanic = ">=19.12.2,<21.0.0"
sanic-cors = "^0.10.0b1"
sanic-jwt = ">=1.3.2,<2.0"
cloudpickle = ">=1.2,<1.7"
aiohttp = ">=3.6,<3.8,!=3.7.4.post0"
questionary = ">=1.5.1,<1.10.0"
prompt-toolkit = "^2.0"
python-socketio = ">=4.4,<6"
python-engineio = ">=4,<6,!=5.0.0"
pydot = "~1.4"
async_generator = "~1.10"
SQLAlchemy = ">=1.3.3,<1.5.0"
sklearn-crfsuite = "~0.3"
psycopg2-binary = ">=2.8.2,<2.10.0"
python-dateutil = "~2.8"
tensorflow = "~2.3"
tensorflow_hub = ">=0.10,<0.13"
tensorflow-addons = ">=0.10,<0.14"
tensorflow-estimator = "~2.3"
tensorflow-probability = ">=0.11,<0.14"
setuptools = ">=41.0.0"
kafka-python = ">=1.4,<3.0"
ujson = ">=1.35,<5.0"
oauth2client = "4.1.3"
regex = ">=2020.6,<2021.8"
joblib = ">=0.15.1,<1.1.0"
sentry-sdk = ">=0.17.0,<1.3.0"
aio-pika = "^6.7.1"
pyTelegramBotAPI = "^3.7.3"
"""
core_requirements = [

]


setuptools.setup(
    name='ruth-nlu',
    description="A Python CLI for Ruth NLP",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    author='Puretalk',
    author_email='info@puretalk.ai',
    version="2.8.0",
    install_requires=core_requirements,
    python_requires='>=3.7,<=3.10',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    package_data={
        "data": ["*.txt"]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={"console_scripts": ["ruth = ruth.__main__:main"]},
)