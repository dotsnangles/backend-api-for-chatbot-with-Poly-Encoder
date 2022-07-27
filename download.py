# download the data and the trained model.

import os, shutil, gdown

def download():
    if os.path.exists('./data'):
        shutil.rmtree('./data')

    if os.path.exists('./model'):
        shutil.rmtree('./model')

    gdown.download_folder(id='1Ipr-aNF5ELMY0HTXAmeV26LlgktKUfmG', quiet=False, use_cookies=False)
    gdown.download_folder(id='1RH7laK4WlucCw68ZeExFvyg7vs-kB_x3', quiet=False, use_cookies=False)
    os.rename('./감성대화챗봇데이터/', './data')
    os.rename('./chatbot_output/', './model')

if __name__ == '__main__':
    download()