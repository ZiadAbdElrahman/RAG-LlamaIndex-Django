from django.http import JsonResponse
from django.shortcuts import render
from .forms import UploadFileForm
from .models import UploadedFile
import requests
from django.conf import settings  # Ensure settings are configured to import correctly
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt  # Exempt from CSRF for simplicity; consider a more secure approach in production

def ai_response(request):
    if request.method == 'POST':
        # Parse the incoming JSON data from the request
        data = json.loads(request.body)
        user_input = data.get('message', '')
        user_id = data.get('user_name', '')

        # Construct the URL for the FastAPI endpoint
        url = f'{settings.AI_ENGINE_URL}/get_response/'

        # Prepare the data to be sent in the POST request
        payload = {'message': user_input, 'user_id':user_id}

        # Send a POST request to the FastAPI endpoint
        try:
            response = requests.post(url, json=payload)
            response_data = response.json()
            return JsonResponse(response_data)
        except requests.exceptions.RequestException as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def upload_file(request):
    # Access user_id from request.POST instead of request.body
    user_id = request.POST.get('user_name')
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
        file = request.FILES['file']
        new_file = UploadedFile(file=file)
        new_file.save()
        file.seek(0)
        response = send_file_to_ai_engine(user_id, file)
        return JsonResponse(response.json(), safe=False)
    
    return JsonResponse({"error": "Invalid form data"}, status=400)

def chat_and_upload_view(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        data = json.loads(request.body)
        user_id = data.get('user_name', '')
        if form.is_valid():
            file = request.FILES['file']
            new_file = UploadedFile(file=request.FILES['file'])
            new_file.save()
            
            return JsonResponse(response.json(), safe=False)
    else:
        form = UploadFileForm()
        user_id = None
        
    return render(request, 'ai_chat/chatbot.html', {'form': form, 'user_id':user_id})

def send_file_to_ai_engine(user_id, file):
    # URL of the FastAPI endpoint
    url = f'{settings.AI_ENGINE_URL}/upload_pdf/'
    files = {'file': (file.name.replace(' ', '-'), file, 'application/pdf')}
    data = {'user_id': user_id}
    try:
        response = requests.post(url, data=data, files=files)
        
        return response
    
    except requests.exceptions.RequestException as e:
        print("Error during file upload:", e)