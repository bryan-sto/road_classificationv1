import os
import subprocess
import json
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt

def predict_images(request):
    json_output = None
    video_output = None

    if request.method == 'POST':
        street_name = request.POST.get('street_name')
        image_folder = request.POST.get('image_folder')  # Add image_folder

        script_path = os.path.join(os.path.dirname(__file__), 'predict.py')

        try:
            subprocess.run(['python', script_path, street_name, image_folder], check=True)
            json_output_path = os.path.join('final_cnn', 'output_folder', f'class_distribution_{street_name}.json')
            video_output_path = os.path.join('final_cnn', 'output_folder', f'generated_video_{street_name}.mp4')
            json_output = read_json_file(json_output_path)
            video_output = video_output_path  # Just the path, no need to read the video
        except subprocess.CalledProcessError:
            pass

    return render(request, 'result_template.html', {'json_output': json_output, 'video_output': video_output})

@api_view(['POST'])
def predict_images_api(request):
    if request.method == 'POST':
        try:
            image_files = request.FILES.getlist('image')  # Use request.FILES.getlist to get a list of uploaded file data
            
            if not image_files:
                return Response({'error': 'Image files are missing'}, status=status.HTTP_400_BAD_REQUEST)
            
            responses = []
            
            for image_data in image_files:
                original_filename = image_data.name
                image_path = os.path.join('final_cnn', 'predict', original_filename)
                
                # Save the image to the predict folder
                with open(image_path, 'wb') as image_file:
                    for chunk in image_data.chunks():
                        image_file.write(chunk)
                
                responses.append({'filename': original_filename, 'message': 'Image uploaded successfully'})
            
            return Response({'results': responses}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data
