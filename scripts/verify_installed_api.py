import argparse, json, os, secrets, socket, subprocess, sys, threading, time
from pathlib import Path
import httpx, uvicorn
import ai_home
from ai_home.api import create_app
from ai_home.config import Settings
parser = argparse.ArgumentParser(description='Verify an installed AI@Home wheel over real loopback HTTP.')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
args = parser.parse_args()
output = args.output_dir.resolve()
assert 'site-packages' in ai_home.__file__, 'Run with the installed wheel and Python -I'
for key in tuple(os.environ):
    if key.startswith(('AI_HOME_', 'SUPABASE_')):
        os.environ.pop(key)
output.mkdir(exist_ok=False)
os.environ['AI_HOME_DATABASE_URL'] = 'sqlite+pysqlite:///' + (output/'api.db').as_posix()
os.environ['AI_HOME_ENVIRONMENT'] = 'test'
for action in ('upgrade', 'check'):
    subprocess.run([sys.executable,'-I','-m','ai_home.migrate',action],check=True)
settings = Settings(_env_file=None, foundation_checkpoint=args.checkpoint.resolve(), foundation_device='cpu', foundation_max_new_tokens=16)
app = create_app(settings)
sock = socket.socket(); sock.bind(('127.0.0.1',0))
port = sock.getsockname()[1]
server = uvicorn.Server(uvicorn.Config(app,log_level='error',access_log=False))
thread = threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True)
thread.start()
try:
    for _ in range(100):
        if server.started: break
        if not thread.is_alive(): raise RuntimeError('Server stopped')
        time.sleep(.05)
    if not server.started: raise RuntimeError('Server failed to start')
    with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=30,trust_env=False) as client:
        payload={'messages':[{'role':'user','content':'Hola'}],'generation':{'max_new_tokens':8,'temperature':0.0}}
        unauth = client.post('/api/v1/chat/completions',json=payload)
        assert unauth.status_code == 401
        credentials={'email':'reproduction@example.com','password':secrets.token_urlsafe(32)}
        response=client.post('/api/v1/auth/register',json=credentials); response.raise_for_status()
        response=client.post('/api/v1/auth/login',json=credentials); response.raise_for_status()
        client.headers['Authorization']='Bearer '+response.json()['access_token']
        response=client.post('/api/v1/chat/completions',json=payload); response.raise_for_status()
        completion=response.json()
        assert completion['choices'][0]['message']['content']
        response=client.post('/api/v1/chat/stream',json=payload); response.raise_for_status()
        events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert any(event.get('type')=='done' for event in events), events
        report={'format':'ai-home-installed-api-smoke-v1','installed_package': 'site-packages' in ai_home.__file__,
                'transport':'real loopback HTTP','schema':'packaged Alembic upgrade + check',
                'unauthenticated_status':unauth.status_code,'completion':completion,'stream_events':events,
                'claim':'Mechanical API inference only; no language quality established'}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('API completion, streaming and authentication verified from installed wheel')
finally:
    server.should_exit=True
    thread.join(timeout=10)
    sock.close()
    app.state.engine.dispose()
    if thread.is_alive(): raise RuntimeError('Server did not stop')
