# MIT 6.5840: Sistemas Distribuidos

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Course](https://img.shields.io/badge/MIT-6.5840-A31F34?style=flat)](http://nil.csail.mit.edu/6.5840/2025/)

## 📋 Descripción

Este repositorio contiene la implementación completa de los laboratorios del curso MIT 6.5840 (anteriormente 6.824), uno de los cursos más prestigiosos sobre sistemas distribuidos. El proyecto implementa desde cero sistemas distribuidos fundamentales incluyendo MapReduce, el algoritmo de consenso Raft, y servicios de almacenamiento clave-valor tolerantes a fallos con particionamiento (sharding).

**¿Por qué es importante?**
- Implementa los algoritmos fundamentales que sustentan sistemas como Google MapReduce, etcd, y Consul
- Proporciona experiencia práctica construyendo sistemas distribuidos reales con tolerancia a fallos
- Incluye un framework de pruebas exhaustivo que simula fallos de red, particiones, y crashes de servidores

## ✨ Características Principales

### Lab 1: MapReduce
- Implementación completa del modelo de programación MapReduce de Google
- Coordinador tolerante a fallos con detección de workers caídos
- Workers distribuidos que procesan tareas Map y Reduce
- Manejo de fallos y redistribución de tareas

### Lab 2: Servidor Key-Value Básico
- Servidor key-value con operaciones Get/Put
- Sincronización con locks para acceso concurrente
- Detección de operaciones duplicadas
- Testing con modelo de consistencia lineal (Porcupine)

### Lab 3: Algoritmo de Consenso Raft (3A-3D)
- **3A**: Elección de líder y heartbeats
- **3B**: Replicación de log y aplicación de comandos
- **3C**: Persistencia del estado de Raft
- **3D**: Compactación de log mediante snapshots

### Lab 4: KVRaft - Servicio Key-Value Tolerante a Fallos (4A-4C)
- Servicio key-value replicado usando Raft
- Máquina de estados replicada (RSM)
- Detección de operaciones duplicadas
- Snapshots para limitar el crecimiento del log
- Garantías de consistencia lineal

### Lab 5: Servicio Key-Value con Particionamiento (5A-5C)
- Sharding dinámico de datos entre grupos de réplicas
- Controlador de configuración de shards
- Migración de datos entre grupos
- Reconfiguración sin tiempo de inactividad
- Alta disponibilidad con múltiples grupos de réplicas

## 🏗 Arquitectura del Sistema

El sistema está organizado en capas, donde cada laboratorio construye sobre el anterior:

```
┌─────────────────────────────────────────────────┐
│         Lab 5: Sharded KV Service               │
│    (Particionamiento + Reconfiguración)         │
├─────────────────────────────────────────────────┤
│         Lab 4: KVRaft Service                   │
│    (Key-Value + Replicación Raft)               │
├─────────────────────────────────────────────────┤
│         Lab 3: Raft Consensus                   │
│    (Líder Election + Log Replication)           │
├─────────────────────────────────────────────────┤
│         Lab 2: Simple KV Service                │
│         Lab 1: MapReduce                        │
└─────────────────────────────────────────────────┘
```

Ver [system-architecture.mmd](system-architecture.mmd) para el diagrama completo de arquitectura.

## 🚀 Instalación

### Prerrequisitos

- **Go 1.22+**: [Descarga Go](https://golang.org/dl/)
- **Git**: Para clonar el repositorio
- **Unix/Linux/macOS**: El proyecto usa sockets Unix

### Pasos de Instalación

1. **Clona el repositorio**:
```bash
git clone https://github.com/tu-usuario/6.5840.git
cd 6.5840
```

2. **Verifica la instalación de Go**:
```bash
go version
# Debería mostrar: go version go1.22 o superior
```

3. **Descarga las dependencias**:
```bash
cd src
go mod download
```

4. **Verifica que todo funciona**:
```bash
cd mr
go test -run Sequential
```

## 💻 Uso

### Lab 1: MapReduce

**Ejecutar el ejemplo secuencial**:
```bash
cd src/main
go build -buildmode=plugin ../mrapps/wc.go
go run mrsequential.go wc.so pg-*.txt
```

**Ejecutar MapReduce distribuido**:
```bash
# Terminal 1: Inicia el coordinador
go run mrcoordinator.go pg-*.txt

# Terminal 2, 3, 4...: Inicia workers
go run mrworker.go wc.so
```

**Ejecutar las pruebas**:
```bash
cd src/main
bash test-mr.sh
```

### Lab 2: Key-Value Server

**Ejecutar las pruebas**:
```bash
cd src/kvsrv1
go test -v
```

**Probar el servidor con locks**:
```bash
cd src/kvsrv1/lock
go test -v
```

### Lab 3: Raft

**Ejecutar todas las pruebas de Raft**:
```bash
cd src/raft1
go test -v
```

**Ejecutar una prueba específica**:
```bash
# Prueba de elección de líder
go test -run 3A

# Prueba de replicación de log
go test -run 3B

# Prueba de persistencia
go test -run 3C

# Prueba de snapshots
go test -run 3D
```

**Ejecutar con race detector**:
```bash
go test -race
```

### Lab 4: KVRaft

**Ejecutar las pruebas**:
```bash
cd src/kvraft1
go test -v
```

**Probar con snapshots**:
```bash
go test -run Snapshot
```

**Ejecutar la máquina de estados replicada**:
```bash
cd src/kvraft1/rsm
go test -v
```

### Lab 5: Sharded KV

**Ejecutar las pruebas**:
```bash
cd src/shardkv1
go test -v
```

**Ejecutar pruebas de reconfiguración**:
```bash
go test -run Challenge
```

## 🔧 Comandos Útiles

### Makefile

El proyecto incluye un Makefile para crear submissions:

```bash
# Crear tarball para Lab 1
make lab1

# Crear tarball para Lab 3A
make lab3a

# Crear tarball para Lab 4B
make lab4b
```

### Testing Exhaustivo

```bash
# Ejecutar una prueba múltiples veces para detectar race conditions
go test -run TestBasic3A -count=100

# Ejecutar con timeout más largo
go test -timeout 30m

# Ejecutar con verbose output y race detector
go test -v -race -timeout 10m
```

### Debugging

Cada módulo incluye utilidades de logging. Para habilitar logs:

```go
// En raft1/util.go
const Debug = true // Cambia a true para ver logs
```

Ejecuta las pruebas y observa los logs:
```bash
go test -run TestBasic3A -v 2>&1 | less
```

## 📊 Diagramas Técnicos

### Arquitectura del Sistema
Ver [system-architecture.mmd](system-architecture.mmd) - Muestra la arquitectura completa desde MapReduce hasta Sharded KV.

### Flujo de Consenso Raft
Ver [sequence-diagram.mmd](sequence-diagram.mmd) - Detalla el proceso de elección de líder y replicación de log en Raft.

### Interfaces Principales
Ver [main-interfaces.mmd](main-interfaces.mmd) - Documenta las interfaces públicas de Raft, RSM, y KVServer.

### Diagrama de Clases
Ver [class-diagram.mmd](class-diagram.mmd) - Muestra las relaciones entre componentes principales.

## 🧪 Framework de Testing

El proyecto incluye un framework de testing sofisticado en `src/tester1/`:

- **config.go**: Configuración de clusters de prueba
- **persister.go**: Simulación de almacenamiento persistente
- **srv.go**: Simulación de servidores
- **clnts.go**: Simulación de clientes
- **annotation.go**: Anotaciones de pruebas

El framework puede simular:
- Fallos de red y particiones
- Crashes y recuperación de servidores
- Pérdida y reordenamiento de mensajes
- Latencia de red variable

## 📂 Estructura del Proyecto

```
6.5840/
├── src/
│   ├── mr/                    # Lab 1: MapReduce
│   │   ├── coordinator.go     # Coordinador de MapReduce
│   │   ├── worker.go          # Worker de MapReduce
│   │   └── rpc.go             # Definiciones RPC
│   │
│   ├── kvsrv1/                # Lab 2: Key-Value Server
│   │   ├── server.go          # Implementación del servidor
│   │   ├── client.go          # Cliente KV
│   │   └── lock/              # Tests con locks
│   │
│   ├── raft1/                 # Lab 3: Raft
│   │   ├── raft.go            # Implementación de Raft
│   │   ├── server.go          # Wrapper del servidor
│   │   └── raft_test.go       # Suite de pruebas
│   │
│   ├── kvraft1/               # Lab 4: KVRaft
│   │   ├── server.go          # Servidor KV con Raft
│   │   ├── client.go          # Cliente KVRaft
│   │   └── rsm/               # Replicated State Machine
│   │       ├── server.go      # Implementación RSM
│   │       └── rsm_test.go    # Pruebas RSM
│   │
│   ├── shardkv1/              # Lab 5: Sharded KV
│   │   ├── server.go          # Servidor con sharding
│   │   ├── client.go          # Cliente con routing
│   │   ├── shardctrler/       # Controlador de shards
│   │   └── shardgrp/          # Grupos de réplicas
│   │
│   ├── labrpc/                # Framework RPC simulado
│   ├── labgob/                # Serialización GOB
│   ├── tester1/               # Framework de testing
│   ├── kvtest1/               # Utilidades de testing KV
│   ├── raftapi/               # Interfaz de Raft
│   ├── models1/               # Modelos de datos
│   └── main/                  # Ejecutables principales
│       ├── mrcoordinator.go   # Coordinador MapReduce
│       ├── mrworker.go        # Worker MapReduce
│       ├── mrsequential.go    # MapReduce secuencial
│       └── test-mr.sh         # Script de pruebas MR
│
├── Makefile                   # Build y submissions
└── .check-build               # Validación de builds
```

## 🔍 Detalles de Implementación

### Raft Consensus

El algoritmo Raft implementa consenso distribuido mediante:

1. **Elección de Líder**: Usa timeouts aleatorios y RequestVote RPC
2. **Replicación de Log**: El líder replica entradas mediante AppendEntries RPC
3. **Seguridad**: Garantiza que logs comprometidos nunca se pierden
4. **Persistencia**: State, log, y votedFor sobreviven crashes
5. **Snapshots**: Compacta logs antiguos para ahorrar espacio

Ver [sequence-diagram.mmd](sequence-diagram.mmd) para el flujo completo.

### Replicated State Machine (RSM)

La capa RSM en Lab 4 proporciona:

- **Linearizabilidad**: Todas las operaciones son linealmente consistentes
- **Detección de Duplicados**: Evita ejecutar la misma operación dos veces
- **Snapshots Automáticos**: Limita el crecimiento del log de Raft
- **Recuperación**: Restaura el estado desde snapshots después de crashes

### Sharding

El sistema de sharding en Lab 5 implementa:

- **Particionamiento Dinámico**: Redistribuye shards entre grupos
- **Migración de Datos**: Transfiere datos sin perder disponibilidad
- **Configuración Distribuida**: Usa un controlador replicado con Raft
- **Leases (Opcional)**: Optimización para lecturas locales

## 🐛 Solución de Problemas

### Problema: Tests fallan intermitentemente
**Solución**: Esto suele indicar race conditions. Ejecuta con `-race`:
```bash
go test -race -run TestFailingTest
```

### Problema: Tests timeout
**Solución**: Aumenta el timeout o verifica que no haya deadlocks:
```bash
go test -timeout 30m
# O añade logs para encontrar dónde se bloquea
```

### Problema: "too many open files"
**Solución**: Aumenta el límite de archivos:
```bash
ulimit -n 4096
```

### Problema: Workers de MapReduce no se conectan
**Solución**: Verifica que no haya sockets Unix huérfanos:
```bash
rm /var/tmp/5840-mr-*
# O reinicia el coordinador
```

### Problema: Mensajes RPC perdidos en tests
**Solución**: Esto es intencional. El framework simula redes no confiables. Tu implementación debe manejar timeouts y reintentos.

## 📚 Recursos de Aprendizaje

### Papers Fundamentales
- [MapReduce (Google, 2004)](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf)
- [Raft Consensus (Stanford, 2014)](https://pdos.csail.mit.edu/6.824/papers/raft-extended.pdf)
- [GFS - Google File System](https://pdos.csail.mit.edu/6.824/papers/gfs.pdf)

### Recursos del Curso
- [Página del Curso MIT 6.5840](http://nil.csail.mit.edu/6.5840/2025/)
- [Lectures y Videos](http://nil.csail.mit.edu/6.5840/2025/schedule.html)
- [Guía de Raft](https://thesquareplanet.com/blog/students-guide-to-raft/)
- [Raft Visualization](https://raft.github.io/)

### Debugging y Testing
- [Effective Go](https://golang.org/doc/effective_go)
- [Go Race Detector](https://go.dev/blog/race-detector)
- [Testing Distributed Systems](https://asatarin.github.io/testing-distributed-systems/)

## 🤝 Contribuir

Este es un proyecto académico del MIT 6.5840. Si eres estudiante del curso:

1. **No copies código**: El honor code del MIT prohíbe compartir soluciones
2. **Trabaja de forma individual**: Implementa tu propia solución
3. **Consulta las reglas del curso**: Revisa la política de colaboración

Si encuentras bugs en el código de testing o infraestructura (no en las soluciones), por favor:
1. Crea un issue describiendo el problema
2. Incluye steps para reproducir
3. Comparte logs relevantes

## 📄 Licencia

Este código está basado en el curso MIT 6.5840. El código de testing e infraestructura es propiedad del MIT. Las implementaciones de laboratorio son trabajo académico.

**Uso Educativo**: Este código es solo para aprendizaje. Si eres estudiante del curso, debes implementar tus propias soluciones.

## 🙏 Agradecimientos

- **MIT PDOS Group**: Por crear y mantener este curso excepcional
- **Robert Morris, Frans Kaashoek, Nickolai Zeldovich**: Profesores del curso
- **Diego Ongaro y John Ousterhout**: Por el algoritmo Raft
- **Google**: Por MapReduce y los papers que inspiraron este curso

## 📞 Contacto y Soporte

- **Página del Curso**: [MIT 6.5840](http://nil.csail.mit.edu/6.5840/2025/)
- **Piazza**: Para estudiantes registrados en el curso
- **Office Hours**: Consulta el schedule del curso

---

**Construido con** ❤️ **para aprender sistemas distribuidos**

*Última actualización: Octubre 2025*
