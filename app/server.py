from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai.vision import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1-2vMyvg8mBOpPAtes3C87IqJJeR3n0WX'
model_file_name = 'eye_first_714.pkl'
classes = ['Normal', 'Diabetic', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia','Other_abnormalities']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    p = learn.predict(img)[2].numpy()
    idx = []
    percent = []
    for i in range(8):
        if p[i]>0.3:
            idx.append(i)
            percent.append(p[i])
    pred = [str(classes[x])+" "+str(percent[x]*100) for x in range(len(idx))]
    pred = " ".join(str(x) for x in pred)
    return JSONResponse({'result': str(pred)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

