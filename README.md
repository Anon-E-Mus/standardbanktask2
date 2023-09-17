# standardbanktask2

Follow https://github.com/edeltech/tensorflow-lite-on-aws-lambda/tree/master to setup the serverless framework on your system.

Copy paste the app.py, Dockerfile, serverless.yml to your project directory to replace the default ones.

Change the bucket names to the ones you have existing.

IAM policies must be assigned manually on AWS IAM itself.

This lambda function needs AWSS3FULLACCESS as well as AWSS3ReadOnlyAccess IAM policies attached to its role.
