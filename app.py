import boto3
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np


s3client = boto3.client('s3')
np.set_printoptions(threshold=np.inf)
filename = ''

def handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    print(event)
    
    print(bucket)
    print(key)
    global filename
    filename = key.split('.')[0] + '.txt'
    
    response = s3client.get_object(Bucket=bucket,Key=key)
    
    data = response['Body']
    
    image = cv2.imdecode(np.asarray(bytearray(data.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    predict_face_in_image(image)
    
    print("Done")



def predict_face_in_image(input_image):
    # Load the MTCNN model for face detection
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load and preprocess the image
    image = input_image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    faceimg = None

    if len(faces) ==1 :
        # If faces are detected, mark them in the image
        for face in faces:
            x, y, width, height = face[0],face[1],face[2],face[3]
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            a_size = max(width,height)
            diff_y = (a_size-height)//2
            diff_x = (a_size-width)//2
            y_d = max((y-diff_y-40),0)

            x_d = max((x-diff_x-40),0)
            a_size = int(a_size*1.3)
            faceimg = image_rgb[y_d:y_d+a_size, x_d:x_d+a_size]
            faceimg = cv2.resize(faceimg, (112,112))

            

            

            

            interpreter = tflite.Interpreter(model_path='mobile_face_net.tflite')
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            print("Number of Layers:", len(interpreter.get_tensor_details()))

            input_shape = input_details[0]['shape']

            
            
            


            #Normalize
            result = faceimg / 255.0
            img_batch = np.expand_dims(result, axis=0)

            

            img32 = img_batch.astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], img32)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data)

            
            print(filename)
            with open(('/tmp/' + filename), 'w') as f:
                f.write(','.join(map(str, output_data[0])))

            upload_to_s3(filename, 'outputsbtask2')


        

        # Return True if faces are detected
        return faceimg
    else:
        # Save the original image without any markings
        print("Multiple or no face detected")

        # Return False if no faces are detected
        return None

def upload_to_s3(filename, bucket):
    s3 = boto3.client('s3')
    s3.upload_file(('/tmp/' + filename), bucket, filename)