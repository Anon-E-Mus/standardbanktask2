FROM public.ecr.aws/lambda/python:3.7

COPY app.py "${LAMBDA_TASK_ROOT}"

COPY mobile_face_net.tflite "${LAMBDA_TASK_ROOT}"

COPY requirements.txt  .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.handler"]
