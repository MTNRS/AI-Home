"""Real Tk registration and worker loop against an isolated loopback API."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import secrets
import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox

from alembic import command
import httpx
import torch
import uvicorn

from ai_home import desktop
from ai_home.api import create_app
from ai_home.config import Settings
from ai_home.credits import CreditLedger
from ai_home.job_credits import user_account_id
from ai_home.migrate import migration_config
from ai_home.models import ComputeNode
from ai_home.foundation.distributed import encode_tensors
from ai_home.foundation.model import FoundationConfig, FoundationModel
from ai_home.foundation.tokenizer import ByteBPE


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--task', choices=('gradient', 'evaluate', 'adapter-gradient'), default='gradient')
parser.add_argument('--require-installed', action='store_true')
args = parser.parse_args()
if args.require_installed:
    assert 'site-packages' in desktop.__file__
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
torch.manual_seed(12)
records = [{"source": "AI@Home authored desktop node fixture", "url": "local:desktop-node",
    "license": "Apache-2.0", "date": "2026-11-26", "language": "es",
    "training_allowed": True, "privacy_reviewed": True, "visibility": "public",
    "text": f"Documento {i}: un modelo aprende con datos autorizados."} for i in range(2)]
tokenizer = ByteBPE.train([r['text'] for r in records], vocab_size=280)
config = FoundationConfig(vocab_size=tokenizer.vocab_size, width=16, layers=1,
                          heads=2, kv_heads=1, hidden=32, context=16)
model = FoundationModel(config)
payload = {'shard': {'records': records, 'tokenizer': tokenizer.to_dict()},
    'config': asdict(config), 'parameters': encode_tensors(dict(model.named_parameters())),
    'torch_version': str(torch.__version__), 'weights_public_and_redistributable': True}
if args.task == 'adapter-gradient':
    from ai_home.foundation.adapters import attach_adapters
    adapted = attach_adapters(model, rank=2, alpha=2.)
    payload = {'base': payload, 'rank': 2, 'alpha': 2.,
        'factors': encode_tensors({k: v for k, v in adapted.named_parameters() if v.requires_grad}),
        'factors_public_and_redistributable': True}
(output / 'payload.json').write_text(json.dumps(payload), encoding='utf-8')
settings = Settings(_env_file=None, environment='test',
    database_url=f"sqlite+pysqlite:///{(output / 'jobs.db').as_posix()}",
    supabase_url=None, supabase_publishable_key='', supabase_jwks_url=None,
    foundation_distributed_training_enabled=args.task == 'gradient',
    foundation_distributed_evaluation_enabled=args.task == 'evaluate',
    foundation_distributed_adapters_enabled=args.task == 'adapter-gradient')
command.upgrade(migration_config(settings), 'head')
api = create_app(settings)
sock = socket.socket()
sock.bind(('127.0.0.1', 0))
url = f'http://127.0.0.1:{sock.getsockname()[1]}'
server = uvicorn.Server(uvicorn.Config(api, log_level='error', access_log=False))
thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
thread.start()
try:
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(.05)
    assert server.started
    with httpx.Client(base_url=url, timeout=30) as client:
        def account(email):
            credentials = {'email': email, 'password': secrets.token_urlsafe(32)}
            response = client.post('/api/v1/auth/register', json=credentials)
            response.raise_for_status()
            user_id = response.json()['user_id']
            response = client.post('/api/v1/auth/login', json=credentials)
            response.raise_for_status()
            return user_id, response.json()['access_token']
        owner_id, owner_token = account('desktop-owner@example.com')
        _, worker_token = account('desktop-worker@example.com')
        ledger = CreditLedger(api.state.engine)
        ledger.create_account('smoke-issuance', 'test_issuance', allow_negative=True)
        owner_account = user_account_id(owner_id)
        ledger.create_account(owner_account, 'user')
        ledger.post(idempotency_key='desktop-smoke-issuance', operation='smoke_issuance',
                    postings={'smoke-issuance': -3, owner_account: 3})
        owner_headers = {'Authorization': f'Bearer {owner_token}'}
        response = client.post('/api/v1/jobs/foundation/' + args.task, headers=owner_headers,
            json={'payload': payload, 'allow_public_node_processing': True,
                  'idempotency_key': 'desktop-' + args.task, 'reward_credits': 3})
        response.raise_for_status()
        job_id = response.json()['job_id']
        # Isolate only local persistence; registration, HTTP, Tk, worker and
        # neural execution use the actual application implementations.
        desktop.load_settings = lambda: {}
        desktop.load_memories = lambda *a: []
        desktop.save_settings = lambda *a: None
        def fail_dialog(*a, **kw):
            raise RuntimeError('Desktop displayed an error during smoke')
        messagebox.showerror = fail_dialog
        credentials = {}
        desktop.keyring.get_password = lambda service, key: credentials.get((service, key))
        desktop.keyring.set_password = lambda service, key, value: credentials.__setitem__((service, key), value)

        def run(window, *unused):
            window.withdraw()
            try:
                window.session_token = worker_token
                window.session_server_url = url
                window.server_var.set(url)
                window.consent_var.set(True)
                assert not window.node_adapters_var.get(), 'Adapters must default to disabled'
                window.node_training_var.set(args.task == 'gradient')
                window.node_evaluation_var.set(args.task == 'evaluate')
                window.node_adapters_var.set(args.task == 'adapter-gradient')
                window.node_device_var.set(args.device)
                window.transfer_probe_var.set(True)
                window._start_contribution()
                deadline = time.monotonic() + 60
                job = None
                while time.monotonic() < deadline:
                    window.update()
                    window._drain_events()
                    response = client.get('/api/v1/jobs', headers=owner_headers)
                    response.raise_for_status()
                    job = next(j for j in response.json() if j['job_id'] == job_id)
                    if job['status'] == 'completed':
                        break
                    time.sleep(.05)
                assert job and job['status'] == 'completed', 'Desktop worker did not complete neural task'
                window._stop_contribution()
                window.node_thread.join(timeout=10)
                assert not window.node_thread.is_alive()
                worker_headers = {'Authorization': f'Bearer {worker_token}'}
                worker_credits = client.get('/api/v1/credits', headers=worker_headers).json()
                owner_credits = client.get('/api/v1/credits', headers=owner_headers).json()
                assert worker_credits['earned'] == 3
                assert owner_credits['available'] == 0 and owner_credits['reserved'] == 0
                with api.state.session_factory() as session:
                    node = session.get(ComputeNode, window.settings['node_id'])
                    assert 'foundation.' + args.task + '.v1' in node.supported_job_types
                    for other in {'gradient', 'evaluate', 'adapter-gradient'} - {args.task}:
                        assert 'foundation.' + other + '.v1' not in node.supported_job_types
                    assert node.hardware['resources']['foundation_device'] == args.device
                    assert 'transfer_measurement' in node.hardware
                    hardware = dict(node.hardware)
                old_thread = window.node_thread
                previous_stop = window.node_stop
                window._switch_memory_profile(url, owner_id)
                window.session_token = owner_token
                window.status_var.set('NEW-PROFILE')
                window._drain_events()
                assert previous_stop.is_set() and window.node_stop is None
                assert window.status_var.get() == 'NEW-PROFILE'
                assert not window.consent_var.get() and not window.node_training_var.get() and not window.node_evaluation_var.get()
                assert not window.node_adapters_var.get()
                errors = []
                messagebox.showerror = lambda *a, **kw: errors.append(a)
                window.consent_var.set(True)
                window._start_contribution()
                deadline = time.monotonic() + 10
                while window.node_registration_active and time.monotonic() < deadline:
                    window.update()
                    window._drain_events()
                    time.sleep(.01)
                assert errors and 'no pertenece' in str(errors)
                assert window.node_thread is old_thread and not old_thread.is_alive()
                report = {'format': 'ai-home-desktop-node-smoke-v1', 'device': args.device,
                    'transport': 'real loopback HTTP', 'job_status': job['status'],
                    'job_type': job['job_type'],
                    'token_count': job['result']['token_count'], 'worker_credits': worker_credits,
                    'owner_credits': owner_credits,
                    'transfer_measurement': hardware['transfer_measurement'],
                    'cross_account_node_reuse_blocked': True,
                    'old_node_status_discarded': True,
                    'claim': 'Real Tk registration and node loop; one host; synthetic accounts and test credits; no general capability claim'}
                (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                (output / 'result.json').write_text(json.dumps(job['result']), encoding='utf-8')
            finally:
                if window.node_stop is not None:
                    window.node_stop.set()
                if window.node_thread is not None:
                    window.node_thread.join(timeout=10)
                window.destroy()
        tk.Tk.mainloop = run
        desktop.main()
finally:
    server.should_exit = True
    thread.join(timeout=10)
    sock.close()
    api.state.engine.dispose()
    assert not thread.is_alive()
print(f'Real desktop node completed a verified Foundation {args.task} task and earned 3 test credits.')
