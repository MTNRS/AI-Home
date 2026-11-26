# Créditos por cómputo verificado

Los créditos son unidades internas sin pagos económicos ni criptomonedas.
Los trabajos Foundation de tokenización y gradientes aceptan `reward_credits`,
entero entre 0 y 1.000.000 (predeterminado 0, contribución voluntaria gratuita).
La recompensa sale del saldo del solicitante; ningún nodo puede emitir créditos
por declarar tiempo, GPU o rendimiento. Los trabajos math.sum.v1 se conservan
sin recompensa para no cambiar su contrato anterior.

## Flujo transaccional

1. Crear trabajo con recompensa reserva su importe en una cuenta escrow propia
   del trabajo. Saldo insuficiente: HTTP 402 y rollback de trabajo/reserva.
2. Repetir la clave idempotente conserva el trabajo. Cambiar payload o importe
   con esa clave devuelve 409; no crea otra reserva.
3. El lease anuncia reward_credits. El servidor verifica resultado y lease;
   solo entonces transfiere el escrow al propietario autenticado del nodo.
4. Estado completado y asientos del ledger se confirman en la misma transacción.
   Un fallo revierte ambos. Repetir la entrega no vuelve a pagar.
5. Cancelación de un trabajo pendiente o agotamiento de leases reembolsa al
   solicitante. La limpieza de leases vencidos ocurre al solicitar otro lease;
   sin polling, la reserva permanece hasta cancelación o siguiente limpieza.

Cada asiento está balanceado y usa la idempotencia del ledger existente.
Cuentas de usuario/escrow no permiten saldos negativos. El usuario no elige
la cuenta destino: procede del propietario del nodo autenticado. Un trabajo
propio ejecutado en un nodo propio no crea saldo neto ni incrementa earned.
Los totales earned son transferencias brutas recibidas de otros usuarios,
no una clasificación de reputación ni una medida de valor económico.

## API autenticada

- `GET /api/v1/credits`: available, reserved, earned de la cuenta propia.
- `GET /api/v1/credits/transactions`: últimos 100 movimientos propios, sin
  exponer cuentas o movimientos ajenos.
- `POST /api/v1/jobs/{job_id}/cancel`: cancela un trabajo pendiente propio y
  devuelve fondos. Repetir cancelación es idempotente. Trabajo ajeno: 404.

Añadir `reward_credits` al JSON de creación de tokenización o gradientes.
No hay endpoint público para emitir créditos. Estos créditos ya permiten
solicitar procesamiento Foundation a terceros; cobro de inferencia, reservas
por streaming y panel de créditos del escritorio siguen pendientes.

## Migración y asignación inicial

Aplicar `python -m alembic upgrade head` antes de actualizar la API.
La migración 0006 añade job_funding y conserva la contabilidad anterior.
No ejecutar downgrade en una base con reservas activas sin conciliación previa.

Un operador con acceso directo a la base puede asignar saldo inicial auditable:

```powershell
python -m ai_home.credit_admin --user-id UUID_EXISTENTE --amount 100 --idempotency-key asignacion-inicial-001 --reason "asignacion inicial"
```

La URL de base se toma de la configuración del backend. El comando exige un
usuario existente, importe acotado, clave idempotente y motivo breve; registra
una transferencia desde system:issuance. Es una asignación administrativa,
no una recompensa por cómputo. No se ha ejecutado contra cuentas reales en
esta sesión. Las claves de base son privilegios de operador, no de clientes.

Al eliminar una cuenta se reembolsan trabajos pendientes antes de borrarlos.
Se conservan ledger y registros de financiación pseudónimos para conciliación;
el saldo retenido no se transfiere a una cuenta nueva automáticamente. No se
promete eliminación de esos registros contables al borrar el perfil.

## Evidencia ejecutada

```powershell
$env:PYTHONPATH="services/api/src"
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
python -m ai_home.foundation.distributed_smoke --output-dir artifacts/distributed-funded-smoke-v2 --reward-credits 5
```

Usar directorio nuevo. Crea una base desechable, un saldo inicial sintético de
20, solicitante y dos usuarios worker independientes. Dos rondas de gradientes
reales generan cuatro liquidaciones: 10 créditos para cada worker y cero
reservado/disponible para el solicitante. El checkpoint final se carga y genera
texto. Evidencia: DISTRIBUTED_FUNDED_SMOKE.json; artefactos originales en
artifacts/distributed-funded-smoke-v1. Todo ocurrió por HTTP loopback en un equipo.

Pruebas: conservación de saldo/asientos, reservas, reintentos, saldo insuficiente,
resultados inválidos, cancelación, leases agotados, rollback tras fallo inyectado,
aislamento de usuarios, eliminación de cuenta, emisión administrativa idempotente
y pago de gradientes reales. PostgreSQL concurrente aún no se ha ejecutado;
SQLite no prueba exclusión de filas o carreras en producción. No se presenta
este módulo como economía de producción ni sistema antifraude completo.


## Desktop account view

The account panel's “Créditos y movimientos” button reads the authenticated
`/api/v1/credits` and `/api/v1/credits/transactions` endpoints. It displays available
and reserved balances, earned credits from other accounts' verified jobs, and up
to 100 recent signed ledger postings. Operator grants, reserves, verified-work
settlements and refunds are distinguished. Empty histories and HTTP/authentication
errors are represented explicitly; no fabricated balance replaces failed reads.

The view is read-only and refreshes on demand. It closes on profile changes;
background callbacks are guarded by both session identity and window existence.
No account identifier can be supplied to read someone else's ledger. Balance and
history are two separate requests and the view says that they may change between
reads. These internal compute credits are not money or cryptocurrency.

Validation: desktop HTTP-contract tests, an actual API/ledger test with funded
owner and separate zero-balance worker, and a hidden Windows Tk rendering/close
check using clearly isolated fixture values. Full suite before the final ledger
bridge test: 199 passed, 4 PostgreSQL skipped; final credits module: 9 passed.
This does not yet implement paid inference, desktop job funding or transfers.
