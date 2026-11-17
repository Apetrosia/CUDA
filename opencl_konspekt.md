# Конспект: OpenCL (Host API) и OpenCL C

> Конспект составлен по двум презентациям: «OpenCL» (хост-код на C) и «OpenCL C» (язык ядер, сравнение с CUDA).
> Пометки **⚠ Ловушка** — места, где на слайдах опечатки или неточности, а на тесте могут спросить «как правильно».
> Пометки **➕ Дополнение** — то, чего нет на слайдах, но что обычно спрашивают / нужно для понимания.

---

# ЧАСТЬ 1. OpenCL: хост-программа (C API)

## 0. Общая картина: что вообще делает хост-программа

OpenCL-программа состоит из двух частей:

1. **Host program** — обычный код на C/C++, выполняется на CPU. Он находит устройства, создаёт контекст и очередь, компилирует ядра, выделяет память, запускает ядра, забирает результат.
2. **Kernel** — функция на языке **OpenCL C**, выполняется на устройстве (GPU, CPU, ускоритель) во множестве экземпляров (work-item).

**Стандартный порядок действий хоста (запомнить как алгоритм!):**

| № | Шаг | Функция |
|---|-----|---------|
| 1 | Получить список платформ (драйверов) | `clGetPlatformIDs` |
| 2 | Получить устройство | `clGetDeviceIDs` |
| 3 | Создать контекст | `clCreateContext` |
| 4 | Создать очередь команд | `clCreateCommandQueue` |
| 5 | Создать программу из исходника | `clCreateProgramWithSource` |
| 6 | Скомпилировать программу | `clBuildProgram` |
| 7 | Получить точку входа (kernel) | `clCreateKernel` |
| 8 | Выделить буферы на устройстве | `clCreateBuffer` |
| 9 | Задать аргументы ядра | `clSetKernelArg` |
| 10 | Поставить ядро в очередь | `clEnqueueNDRangeKernel` |
| 11 | Дождаться завершения | `clFinish` / `clWaitForEvents` |
| 12 | Прочитать результат | `clEnqueueReadBuffer` |

Заголовок: `#include <CL/cl.h>`.

---

## 1. Получение списка платформ (драйверов)

```c
#include <CL/cl.h>
int main()
{
    cl_platform_id platform[10];
    cl_uint num_platforms;
    clGetPlatformIDs(10, platform, &num_platforms);
```

**Прототип:**

```c
cl_int clGetPlatformIDs(cl_uint numEntries,
                        cl_platform_id *platforms,
                        cl_uint *numPlatforms);
```

| Параметр | Смысл |
|----------|-------|
| `numEntries` | размер массива, на который указывает `platforms` (**0**, если `platforms == NULL`) |
| `platforms` | массив для возврата информации о драйверах (**NULL** — не возвращать) |
| `numPlatforms` | возвращаемое количество драйверов OpenCL (**NULL** — не возвращать) |

**Что важно понимать:**
- **Платформа = драйвер (реализация OpenCL)** от конкретного производителя (NVIDIA, AMD, Intel и т.д.). В системе их может быть несколько.
- Типичный приём «двух вызовов»: сначала `clGetPlatformIDs(0, NULL, &n)` — узнать количество, потом выделить массив на `n` и вызвать ещё раз. На слайде упрощённо: массив на 10 элементов.
- `cl_platform_id` — непрозрачный дескриптор (opaque handle).
- Возвращаемое значение `cl_int` — код ошибки (`CL_SUCCESS` = 0 при успехе).

---

## 2. Обнаружение устройств

```c
cl_device_id device;
cl_uint num_gpgpu;
clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &device, &num_gpgpu);
if (num_gpgpu == 0) {
    clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
}
```

**Логика примера:** сначала пытаемся взять **GPU** с первой платформы; если GPU нет (`num_gpgpu == 0`) — берём **CPU**. Это типичный fallback.

**Прототип:**

```c
cl_int clGetDeviceIDs(cl_platform_id platformID,
                      cl_device_type nDeviceType,
                      cl_uint nNumEntries,
                      cl_device_id *pDevices,
                      cl_uint *pnNumDevices);
```

**Типы устройств (`nDeviceType`):**

| Константа | Значение |
|-----------|----------|
| `CL_DEVICE_TYPE_CPU` | центральный процессор |
| `CL_DEVICE_TYPE_GPU` | видеокарта |
| `CL_DEVICE_TYPE_ACCELERATOR` | специализированный ускоритель |
| `CL_DEVICE_TYPE_DEFAULT` | устройство по умолчанию в системе |
| `CL_DEVICE_TYPE_ALL` | все доступные устройства OpenCL |

> ⚠ Ловушка: на слайде написано «по **молчанию**» — это опечатка, имеется в виду «по умолчанию».

➕ Дополнение: если подходящих устройств нет, функция вернёт ошибку `CL_DEVICE_NOT_FOUND` (поэтому в реальном коде проверяют код возврата). В примере с слайда проверка идёт через счётчик `num_gpgpu`.

---

## 3. Создание контекста и очереди команд

```c
cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
```

### 3.1. Контекст

```c
cl_context clCreateContext(const cl_context_properties *properties,
                           cl_uint num_devices,
                           const cl_device_id *devices,
                           void (CL_CALLBACK *pfnNotify)(...),
                           void *userData,
                           cl_int *errcode_ret);
```

- **Контекст** — «окружение», в котором живут устройства, буферы (память), программы, ядра и очереди. Объекты памяти и программы принадлежат контексту.
- `properties` — свойства контекста (`NULL` — по умолчанию).
- `num_devices`, `devices` — сколько и какие устройства входят в контекст.
- `pfnNotify` — **функция обратного вызова (callback)** для сообщений об ошибках, возникающих внутри контекста во время работы. Её сигнатура:

```c
void CL_CALLBACK pfnNotify(const char *errinfo,
                           const void *private_info,
                           size_t cb,
                           void *user_data);
```

  - `errinfo` — строка с описанием ошибки;
  - `private_info`, `cb` — бинарные данные реализации и их размер;
  - `user_data` — то, что передали в `userData`.
- `userData` — пользовательские данные, передаваемые в callback.
- `errcode_ret` — сюда записывается код ошибки (можно `NULL`).

### 3.2. Очередь команд

```c
cl_command_queue clCreateCommandQueue(cl_context context,
                                      cl_device_id device,
                                      cl_command_queue_properties properties,
                                      cl_int *errcode_ret);
```

- **Очередь команд (command queue)** — через неё хост отправляет устройству команды: запуск ядра, копирование памяти и т.д. **Одна очередь привязана к одному устройству.**
- `properties` — «список свойств очереди команд: можно ли выполнять команды **не последовательно** и разрешено ли **профилирование** команд». То есть:
  - `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` — разрешить выполнение команд не в порядке постановки (out-of-order);
  - `CL_QUEUE_PROFILING_ENABLE` — разрешить профилирование (замеры времени).
  - `0` — по умолчанию: порядок последовательный (in-order), профилирование выключено.

> ⚠ Ловушка (важно!): в примере очередь создаётся с `properties = 0`, но дальше в слайдах про замеры времени используется `clGetEventProfilingInfo`. Чтобы профилирование **работало**, очередь нужно создавать с флагом `CL_QUEUE_PROFILING_ENABLE`. Иначе `clGetEventProfilingInfo` вернёт ошибку `CL_PROFILING_INFO_NOT_AVAILABLE`.

➕ Дополнение: в OpenCL 2.0+ `clCreateCommandQueue` объявлена устаревшей (deprecated), вместо неё — `clCreateCommandQueueWithProperties`.

---

## 4. Компиляция программы

```c
char *source = /* читаем из файла код kernel-функции */;
cl_program program = clCreateProgramWithSource(context, 1,
                         (const char **)&source, NULL, NULL);
cl_int errcode = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
```

### 4.1. `clCreateProgramWithSource`

```c
cl_program clCreateProgramWithSource(cl_context context,
                                     cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     cl_int *errcode_ret);
```

- Создаёт **объект программы** из **исходного текста** (строки) на OpenCL C.
- `count` — число строк в массиве `strings`.
- `strings` — массив указателей на строки с кодом.
- `lengths` — длины строк (`NULL` — строки заканчиваются нулём `\0`).
- Исходник обычно **читается из файла** в память хоста (то есть ядро хранится в отдельном `.cl` файле или строкой в коде).

### 4.2. `clBuildProgram`

```c
cl_int clBuildProgram(cl_program program,
                      cl_uint num_devices,
                      const cl_device_id *device_list,
                      const char *options,
                      void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
                      void *user_data);
```

- **Компилирует и линкует** программу для указанных устройств (**компиляция происходит во время выполнения хост-программы** — JIT, а не заранее).
- `options` — опции компилятора (например `-D`, `-cl-fast-relaxed-math`), `NULL` — без опций.
- `pfn_notify` — **callback по завершении сборки**: `void CL_CALLBACK pfn_notify(cl_program program, void *user_data)`. Если callback задан — сборка **асинхронная**; если `NULL` — функция ждёт завершения.
- Возвращает код ошибки; при ошибке компиляции лог можно получить через `clGetProgramBuildInfo` (➕).

---

## 5. Определение точки входа: создание объекта kernel

```c
cl_kernel kernel = clCreateKernel(program, "kernel_function_name", NULL);
```

```c
cl_kernel clCreateKernel(cl_program program,
                         const char *kernel_name,
                         cl_int *errcode_ret);
```

- Из программы (которая может содержать много функций) выбираем **конкретную kernel-функцию по имени** (строка!).
- Имя должно совпадать с именем функции, помеченной `__kernel`.

**Пример ядра со слайда (сумма элементов массива):**

```c
__kernel void sum(__global int *x, int n, __global int *res)
{
    uint i, j, id = get_global_id(0), sz = get_global_size(0);
    int s = 0;
    for (int i = id; i < n; i += sz)
        s += x[i];
    atomic_add(&res[0], s);
}
```

**Разбор:**
- `__kernel` — это функция-точка входа, вызываемая с хоста.
- `__global int *x` — указатель на данные в **глобальной памяти** устройства.
- `get_global_id(0)` — глобальный номер текущего work-item (по измерению 0).
- `get_global_size(0)` — общее число work-item (по измерению 0).
- **Цикл с шагом `sz`** — каждый work-item обрабатывает элементы `id, id+sz, id+2*sz, …` (так можно обработать массив любого размера `n` независимо от числа потоков; это «grid-stride loop»).
- Каждый work-item считает **частичную сумму `s`** в приватной переменной.
- `atomic_add(&res[0], s)` — **атомарное** сложение частичной суммы в общий результат (без атомарности была бы гонка данных — race condition).

> ⚠ Ловушка: в коде есть лишние `uint i, j` и ещё `int i` в цикле (затенение переменной) — на логику не влияет. В OpenCL 1.0 атомарные функции для 32-битных целых имели имя `atom_add` (с расширением), в 1.1+ — `atomic_add`.

---

## 6. Выделение памяти на ускорителе

```c
cl_mem x_gpu = clCreateBuffer(context,
                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                  n * sizeof(x[0]), x, NULL);
```

```c
cl_mem clCreateBuffer(cl_context context,
                      cl_mem_flags flags,
                      size_t size,
                      void *host_ptr,
                      cl_int *errcode_ret);
```

| Параметр | Смысл |
|----------|-------|
| `context` | контекст, которому принадлежит буфер |
| `flags` | флаги доступа и способа размещения |
| `size` | размер в **байтах** |
| `host_ptr` | указатель на данные хоста (нужен для флагов `USE_HOST_PTR` / `COPY_HOST_PTR`), иначе `NULL` |

**Флаги `cl_mem_flags`:**

| Флаг | Значение |
|------|----------|
| `CL_MEM_READ_WRITE` | чтение / запись (ядро и читает, и пишет) |
| `CL_MEM_WRITE_ONLY` | только запись (со стороны ядра) |
| `CL_MEM_READ_ONLY` | только **чтение** (со стороны ядра) |
| `CL_MEM_USE_HOST_PTR` | использовать память в CPU напрямую (буфер «живёт» в памяти хоста) |
| `CL_MEM_ALLOC_HOST_PTR` | выделить память в CPU (в памяти хоста, доступной устройству) |
| `CL_MEM_COPY_HOST_PTR` | **скопировать** данные из CPU в GPU при создании буфера |

> ⚠ Ловушка: на слайде у `CL_MEM_READ_ONLY` написано «только запись» — **опечатка**. Правильно: **только чтение**.

**Нюансы:**
- Флаги доступа (`READ_WRITE`, `WRITE_ONLY`, `READ_ONLY`) можно комбинировать с флагами размещения через `|` (как в примере: `READ_WRITE | COPY_HOST_PTR`).
- `COPY_HOST_PTR` требует ненулевого `host_ptr` — оттуда данные копируются в буфер.
- `cl_mem` — дескриптор объекта памяти (аналог `cudaMalloc`-указателя, но это не обычный указатель, на хосте разыменовывать его нельзя).

---

## 7. Инициализация аргументов kernel-функции

```c
clSetKernelArg(kernel, 0, sizeof(x_gpu), (void*)&x_gpu);
clSetKernelArg(kernel, 1, sizeof(n),     (void*)&n);
```

```c
cl_int clSetKernelArg(cl_kernel kernel,
                      cl_uint arg_index,
                      size_t arg_size,
                      const void *arg_value);
```

- `arg_index` — **порядковый номер параметра** (считается с **0**), соответствует позиции в сигнатуре kernel-функции.
- `arg_size` — размер аргумента в байтах.
- `arg_value` — **указатель** на значение аргумента (поэтому передаём `&x_gpu`, `&n`).
- Для буфера передаётся сам `cl_mem` (через указатель на него), для скаляра — значение.

➕ Дополнение: для `__local` аргумента передают размер и `NULL`: `clSetKernelArg(kernel, i, size, NULL);` — так выделяется локальная память заданного размера.

---

## 8. Добавление задачи в очередь (запуск ядра)

```c
size_t gridSize = 10;
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gridSize, NULL, 0, NULL, NULL);
clFinish(queue);
// вариант с событием:
// clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gridSize, NULL, 0, NULL, &event);
// clWaitForEvents(1, &event);
```

```c
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                              cl_kernel kernel,
                              cl_uint work_dim,
                              const size_t *global_work_offset,
                              const size_t *global_work_size,
                              const size_t *local_work_size,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event);
```

| Параметр | Смысл |
|----------|-------|
| `command_queue` | очередь, в которую ставим команду |
| `kernel` | запускаемое ядро |
| `work_dim` | **размерность пространства work-item** (1, 2 или 3) |
| `global_work_offset` | смещение глобальных индексов (`NULL` = 0) |
| `global_work_size` | массив: общее число work-item по каждому измерению |
| `local_work_size` | массив: размер work-group по каждому измерению (`NULL` — выбор отдаётся реализации) |
| `num_events_in_wait_list`, `event_wait_list` | список событий, которых нужно дождаться **до** старта этой команды (зависимости) |
| `event` | возвращаемое событие этой команды (для ожидания/профилирования); `NULL` — не нужно |

**Ключевое:**
- Запуск **асинхронный**: функция лишь **ставит** команду в очередь и сразу возвращает управление хосту.
- Чтобы дождаться завершения:
  - `clFinish(queue)` — блокирует хост, пока **все** команды в очереди не выполнены;
  - `clWaitForEvents(1, &event)` — ждёт конкретное событие.
- В примере `gridSize = 10` — запускается **10 work-item** в одномерном пространстве, `local_work_size = NULL`.

➕ Дополнение: если `local_work_size` задан, то (до OpenCL 2.0) `global_work_size` **должен нацело делиться** на `local_work_size` по каждому измерению. Аналог CUDA: `kernel<<<grid, block>>>`, но в CUDA задают число блоков, а в OpenCL — **общее число work-item** (`global_work_size`), а размер группы — отдельно.

---

## 9. Копирование данных из памяти ускорителя на хост

```c
clEnqueueReadBuffer(queue, res_gpu, CL_TRUE, 0, res_size, res_cpu, 0, NULL, NULL);
```

```c
cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                           cl_mem buffer,
                           cl_bool blocking_read,
                           size_t offset,
                           size_t cb,
                           void *ptr,
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event);
```

| Параметр | Смысл |
|----------|-------|
| `buffer` | откуда читаем (буфер на устройстве) |
| `blocking_read` | `CL_TRUE` — **блокирующее** чтение (функция вернётся, когда данные уже в `ptr`); `CL_FALSE` — асинхронное |
| `offset` | смещение в буфере (в байтах) |
| `cb` | сколько байт копировать |
| `ptr` | куда копировать (память хоста) |
| остальное | как у `clEnqueueNDRangeKernel` — события ожидания и результирующее событие |

- При `CL_TRUE` отдельный `clFinish` не обязателен — чтение дождётся выполнения предыдущих команд (в in-order очереди).
- Обратная операция — `clEnqueueWriteBuffer` (➕).

---

## 10. Проверка ошибок

```c
#define checkError(func) \
if (errcode != CL_SUCCESS){\
    printf("Error in " #func "\nError code = %d\n", errcode);\
    exit(1);\
}
#define checkErrorEx(command) \
command; \
checkError(command);
```

**Как это работает:**
- Почти все функции OpenCL возвращают код ошибки (`cl_int`) либо записывают его через последний параметр `errcode_ret`.
- `CL_SUCCESS` (= 0) — всё хорошо.
- Макрос `checkError(func)` проверяет глобальную переменную `errcode`, и при ошибке печатает имя функции и код и завершает программу.
- `#func` — оператор **строкизации** препроцессора: превращает аргумент макроса в строку.
- `checkErrorEx(command)` — сначала **выполняет команду** (например `errcode = clXxx(...)`), потом проверяет `errcode`.

> ⚠ Нюанс: макрос `checkErrorEx` вычисляет `command` и затем подставляет его же как текст в строку для печати (`#func`), но повторно не выполняет. Переменная `errcode` должна быть объявлена выше.

---

## 11. Замеры времени (профилирование)

```c
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gridSize, NULL, 0, NULL, &event);
clWaitForEvents(1, &event);
checkErrorEx(errcode = clGetEventProfilingInfo(event,
        CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
checkErrorEx(errcode = clGetEventProfilingInfo(event,
        CL_PROFILING_COMMAND_END,   sizeof(time_end),   &time_end,   NULL));
tgpu = (double)(time_end - time_start) / 1e9;
```

```c
cl_int clGetEventProfilingInfo(cl_event event,
                               cl_profiling_info param_name,
                               size_t param_value_size,
                               void *param_value,
                               size_t *param_value_size_ret);
```

**Алгоритм:**
1. Запускаем ядро, **получая событие** `event` (последний параметр не `NULL`).
2. Ждём завершения: `clWaitForEvents`.
3. Запрашиваем временные метки:
   - `CL_PROFILING_COMMAND_START` — время начала выполнения команды на устройстве;
   - `CL_PROFILING_COMMAND_END` — время окончания.
4. Разность `end − start` — время в **наносекундах**, поэтому деление на `1e9` даёт **секунды**.

**Важно:**
- Типы `time_start`, `time_end` — `cl_ulong` (64 бита).
- Очередь должна быть создана с `CL_QUEUE_PROFILING_ENABLE`.
- ➕ Есть и другие метки: `CL_PROFILING_COMMAND_QUEUED`, `CL_PROFILING_COMMAND_SUBMIT`.
- Этот способ измеряет **время работы на устройстве**, без накладных расходов хоста (в отличие от замера через `clock()` вокруг вызова).

---

## 12. Сборка программы

**Если CUDA SDK устанавливался с сайта NVIDIA:**

```bash
gcc -std=c99 -L/usr/local/cuda/lib64 -I/usr/local/cuda/include main.c -lOpenCL -O2 -o proga
```

**Если из репозитория (пакеты дистрибутива):**

```bash
gcc -std=c99 main.c -lOpenCL -O2 -o proga
g++ -std=c++11 main.cpp -lOpenCL -O2 -o proga
```

**Расшифровка флагов:**
- `-std=c99` / `-std=c++11` — стандарт языка;
- `-L<путь>` — где искать библиотеки (`lib64`);
- `-I<путь>` — где искать заголовки (`CL/cl.h`);
- `-lOpenCL` — подключить библиотеку OpenCL (`libOpenCL.so`);
- `-O2` — оптимизация;
- `-o proga` — имя выходного файла.

---

# ЧАСТЬ 2. OpenCL C и сравнение с CUDA

## 13. Соответствие терминов CUDA ↔ OpenCL

| CUDA | OpenCL |
|------|--------|
| Kernel | Kernel |
| Host program | Host program |
| **Thread** | **Work item** |
| **Block** | **Work group** |
| **Grid** | **NDRange** |
| **Warp** | **Wavefront** |

- **Work item** — один экземпляр ядра (минимальная единица выполнения).
- **Work group** — группа work-item, которые могут **синхронизироваться** (барьер) и иметь **общую локальную память**.
- **NDRange** — всё индексное пространство (N-мерное, N = 1, 2, 3), «Index Space».
- **Wavefront/Warp** — аппаратная группа потоков, выполняемых в lock-step (SIMT). Warp = 32 у NVIDIA, wavefront = 64 (исторически) у AMD. (➕ числа — для справки.)

---

## 14. Соответствие встроенных функций индексации

| CUDA | OpenCL |
|------|--------|
| `threadIdx.x` | `get_local_id(0)` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `get_global_id(0)` |
| `gridDim.x` | `get_num_groups(0)` |
| `blockIdx.x` | `get_group_id(0)` |
| `blockDim.x` | `get_local_size(0)` |
| `gridDim.x * blockDim.x` | `get_global_size(0)` |

**Главная формула:**

```
get_global_id(0) = get_group_id(0) * get_local_size(0) + get_local_id(0)
get_global_size(0) = get_num_groups(0) * get_local_size(0)
```

Аргумент `0` — номер **измерения** (0 = x, 1 = y, 2 = z). Для 2D: `get_global_id(1)` — это y-координата.

> ⚠ Ловушка: в CUDA `gridDim` — число **блоков** в сетке, а `get_global_size` в OpenCL — число **work-item**. Поэтому прямое соответствие `gridDim.x ↔ get_num_groups(0)` (число групп), а `gridDim.x * blockDim.x ↔ get_global_size(0)`.

---

## 15. Модель исполнения

### CUDA
- Хост запускает **Kernel 1** → на устройстве создаётся **Grid 1**; **Kernel 2** → **Grid 2**.
- Grid состоит из **блоков** (Block (0,0), (1,0), (2,0), (0,1), …), блок — из **потоков** (Thread (0,0)…(4,2)).
- Блок и поток имеют 2D (и даже 3D) индексы.

### OpenCL
- **Index Space (NDRange)** состоит из **Work Group** (индексы (0,0), (1,0), (2,0), (0,1), …).
- Каждая Work Group состоит из **Work Item** (индексы (0,0)…(4,2)).
- На слайде показаны размеры:
  - `get_global_size(0)`, `get_global_size(1)` — размеры всего пространства по осям;
  - `get_local_size(0)`, `get_local_size(1)` — размеры work group по осям.

Иерархия: **NDRange → Work Group → Work Item** (то же, что **Grid → Block → Thread**).

---

## 16. Модель памяти

### CUDA
- **Global Memory** — доступна всем потокам и хосту.
- **Constant Memory** — доступна всем потокам (только чтение), хост её заполняет.
- **Shared Memory** — одна на **блок**, доступна всем потокам блока.
- **Registers / Local memory** — приватные для потока.

### OpenCL
- **Global/Constant Memory** — на устройстве, обмен с **Host Memory** идёт через API.
- **Local Memory** — одна на **work group**, общая для всех work-item группы.
- **Private Memory** — собственная память каждого work-item.

### Таблица соответствия

| CUDA | OpenCL |
|------|--------|
| Global memory | Global memory |
| Constant memory | Constant memory |
| **Shared memory** | **Local memory** |
| **Local memory** | **Private memory** |

> ⚠ Ловушка (классика для теста!): слово **Local** в CUDA и в OpenCL означает **разное**.
> - CUDA **Shared** = OpenCL **Local** (общая для группы).
> - CUDA **Local** = OpenCL **Private** (приватная для потока).

**Квалификаторы адресных пространств в OpenCL C:** `__global`, `__constant`, `__local`, `__private` (по умолчанию переменные внутри функции — `__private`).

---

## 17. Синхронизация

| CUDA | OpenCL |
|------|--------|
| `__syncthreads()` | `barrier()` |

- Барьер синхронизирует **work-item внутри одной work group**. Между разными группами синхронизации нет.
- В OpenCL `barrier` принимает флаг: `barrier(CLK_LOCAL_MEM_FENCE)` (гарантия видимости записей в локальную память) или `barrier(CLK_GLOBAL_MEM_FENCE)`.
- ➕ Барьер должен выполняться **всеми** work-item группы (нельзя ставить в ветке, куда часть потоков не зайдёт) — иначе зависание/неопределённое поведение.

---

## 18. Язык OpenCL C

**Это:**
- **Подмножество ISO C99** — *без* некоторых возможностей C99:
  - стандартных заголовков C99,
  - **указателей на функции**,
  - **рекурсии**,
  - **массивов переменной длины (VLA)**,
  - **битовых полей**.
- **Надмножество C99** — с добавлениями для:
  - **work-items и workgroups** (функции индексации),
  - **векторных типов**,
  - **синхронизации**,
  - **квалификаторов адресных пространств**.
- Также включает большой набор **встроенных функций**:
  - работа с изображениями (image manipulation),
  - работа с work-item,
  - специальные математические функции и т.д.

*(Слайд Khronos Group, 2010.)*

---

## 19. Пример: одно и то же ядро на CUDA и OpenCL

**CUDA:**

```c
__global__ void sum(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
```

**OpenCL:**

```c
__kernel void vecAdd(__global float *a, __global float *b, __global float *c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
```

**Различия:**
1. `__global__` (CUDA) → `__kernel` (OpenCL).
2. Указатели в OpenCL нужно снабжать **квалификатором адресного пространства** (`__global float *a`).
3. Индекс вычисляется одной функцией `get_global_id(0)`.

> ➕ В обоих вариантах нет проверки выхода за границы; если число потоков больше размера массива, нужна проверка `if (i < n)`.

---

## 20. Таблица квалификаторов: перенос CUDA → OpenCL (слайд AMD)

| C for CUDA | OpenCL |
|------------|--------|
| `__global__` function | `__kernel` function |
| `__device__` function | function (**квалификатор не нужен**) |
| `__constant__` variable declaration | `__constant` variable declaration |
| `__device__` variable declaration | `__global` variable declaration |
| `__shared__` variable declaration | `__local` variable declaration |

Обратить внимание: в OpenCL квалификаторы пишутся с **двумя подчёркиваниями в начале и без подчёркиваний в конце** (`__kernel`, `__global`), в CUDA — с двух сторон (`__global__`). Допускаются и варианты без подчёркиваний (`kernel`, `global`, `local`, `constant`).

---

## 21. Типы данных

| Скалярный тип | Векторный тип (n = 2, 4, 8, 16) | Тип для хоста (API) |
|---------------|---------------------------------|---------------------|
| `char`, `uchar` | `charn`, `ucharn` | `cl_char<n>`, `cl_uchar<n>` |
| `short`, `ushort` | `shortn`, `ushortn` | `cl_short<n>`, `cl_ushort<n>` |
| `int`, `uint` | `intn`, `uintn` | `cl_int<n>`, `cl_uint<n>` |
| `long`, `ulong` | `longn`, `ulongn` | `cl_long<n>`, `cl_ulong<n>` |
| `float` | `floatn` | `cl_float<n>` |

- Допустимые размеры вектора: **2, 4, 8, 16** (в OpenCL 1.1+ ещё и 3).
- Типы с `u` — беззнаковые.
- На хосте для передачи векторов в ядро используют типы `cl_*`, чтобы совпадали размер и выравнивание.
- Размеры скаляров фиксированы стандартом (в отличие от C): `char` — 8 бит, `short` — 16, `int` — 32, `long` — 64.

---

## 22. Доступ к компонентам вектора

### 22.1. Для векторов из 2 и 4 компонент: `.x .y .z .w`

```c
float2 pos;
pos.x = 1.0f;
pos.y = 1.0f;
pos.z = 1.0f;   // ОШИБКА: у вектора всего 2 компоненты

float4 c;
c.x = 1.0f;
c.y = 1.0f;
c.z = 1.0f;
c.w = 1.0f;
```

- Для `<vector2>`: `.xy`; для `<vector4>`: `.xyzw`.
- Обращение к несуществующей компоненте (`pos.z` у `float2`) — **ошибка компиляции**.

### 22.2. Числовые индексы: `.s0 … .sF`

| Размер вектора | Допустимые индексы |
|----------------|--------------------|
| 2 компоненты | 0, 1 |
| 4 компоненты | 0, 1, 2, 3 |
| 8 компонент | 0 … 7 |
| 16 компонент | 0 … 9, a, A, b, B, c, C, d, D, e, E, f, F |

```c
float8 f;
f.s0 = 1.0f;   // 1-я компонента
f.s7 = 1.0f;   // 8-я компонента

float16 x;
x.sa = 1.0f;   // или x.sA — 10-я компонента
x.sF = 1.0f;   // или x.sF — 16-я компонента
```

- Префикс **`s`** (или `S`) + шестнадцатеричная цифра.
- Нумерация с **нуля**: `s0` — первая, `sF` — шестнадцатая.

> ⚠ Ловушка: в таблице на слайде для 16 компонент пропущены `d, D` (написано «…c, C, e, E, f, F») — это опечатка. Всего 16 индексов: `0–9` и `a–f` (в обоих регистрах).
> Также в примере на слайде вектор объявлен как `float16 x;`, а обращаются к `f.sa` / `f.sF` — это ошибка в примере, должно быть `x.sa`, `x.sF`.

### 22.3. Удобные суффиксы: `.lo .hi .odd .even`

| Суффикс | Возвращает |
|---------|------------|
| `.lo` | **нижнюю** половину вектора |
| `.hi` | **верхнюю** половину вектора |
| `.odd` | **нечётные** компоненты |
| `.even` | **чётные** компоненты |

```c
float4 f = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
float2 low, high;
float2 o, e;

low  = f.lo;    // f.xy → (1.0f, 2.0f)
high = f.hi;    // f.zw → (3.0f, 4.0f)
o    = f.odd;   // f.yw → (2.0f, 4.0f)
e    = f.even;  // f.xz → (1.0f, 3.0f)
```

**Запомнить:** нумерация с нуля, поэтому «нечётные» — компоненты с индексами 1 и 3 (это `y` и `w`), «чётные» — с индексами 0 и 2 (`x` и `z`).

---

## 23. Ещё несколько примеров (swizzle и векторные литералы)

```c
float4 c;
c.xyzw = (float4)(1.0f, 2.0f, 3.0f, 4.0f);   // присваивание всех компонент

float4 pos = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
float4 swiz = pos.wzyx;     // разворот: (4, 3, 2, 1)   — перестановка (swizzle)
float4 dup  = pos.xxyy;     // повтор:   (1, 1, 2, 2)   — дублирование компонент
pos.xw = (float2)(5.0f, 6.0f);   // pos станет (5, 2, 3, 6)

float4 a, b, c, d;
float16 x;
x = (float16)(a.xxxx, b.xyz, c.xyz, d.xyz, a.yzw);   // 4+3+3+3+3 = 16
```

**Что здесь важно:**
- **Векторный литерал**: `(float4)(1.0f, 2.0f, 3.0f, 4.0f)` — синтаксис приведения типа с круглыми скобками со списком значений.
- **Swizzle** — произвольная перестановка/повтор компонент при **чтении** (`pos.wzyx`, `pos.xxyy`).
- При **записи** (слева от `=`) нельзя повторять компоненты: `pos.xx = ...` — ошибка. Можно любую выборку без повторов: `pos.xw = (float2)(...)`.
- Размеры должны совпадать: справа вектор из стольких же компонент, сколько выбрано слева.
- Можно собирать большие векторы из частей: сумма размеров всех фрагментов должна быть равна размеру итогового вектора (`4+3+3+3+3 = 16`).

---

## 24. Математические функции

| Функция | Смысл |
|---------|-------|
| `sin, cos, tan, sinh, cosh, tanh` | тригонометрические и гиперболические |
| обратные к перечисленным | **вначале добавить букву `a`**: `asin, acos, atan, asinh, acosh, atanh` |
| `gentype sincos(gentype x, gentype *cosval)` | вычисляет **сразу и синус, и косинус** (возвращает sin, а cos записывает по указателю) |
| `atan2(y, x)` | арктангенс `y/x` с учётом квадранта; `atanpi` = arctg / π; также есть варианты `...pi` для других (`asinpi`, `acospi`, `sinpi`, `cospi`, `tanpi`) |
| `sqrt`, `cbrt` | квадратный, кубический корень |
| `rootn(x, y)` | `x^(1/y)` (корень y-й степени, y — целое) |
| `erf(z)` | интеграл ошибок (функция ошибок) |
| `exp, exp2, exp10` | `e^x`, `2^x`, `10^x` |
| `expm1` | `e^x − 1` (точнее при малых x) |
| `log, log2, log10` | логарифмы (натуральный, по основанию 2 и 10) |
| `ceil, floor, round` | округление (вверх, вниз, к ближайшему) |
| `fabs, fmax(x,y), fmin(x,y)` | модуль, максимум, минимум — **поэлементно** (для векторов) |
| `hypot(x, y)` | `sqrt(x² + y²)` |
| `mad(a, b, c)` | `a*b + c` (multiply-add, возможно с меньшей точностью/быстрее) |
| `pow(x, y)`, `pown(x, n)` | степень; `pown` — целая степень `n` |
| `remainder(x, y)` | остаток от деления x на y |
| `clamp(x, minval, maxval)` | `min(max(x, minval), maxval)` — ограничение значения диапазоном |
| `degrees(radians)`, `radians(degrees)` | перевод радианы → градусы и градусы → радианы |
| `gentype step(gentype edge, gentype x)` | **функция Хевисайда**: 0, если `x < edge`, иначе 1 |
| `smoothstep(edge0, edge1, x)` | **сглаженный Хевисайд**: 0 при `x ≤ edge0`, 1 при `x ≥ edge1`, плавный переход между |
| `gentype sign(gentype x)` | знак числа (−1, 0 или +1) |

**Замечания:**
- `gentype` — «обобщённый тип»: функция работает как со скаляром `float`, так и с векторами `floatn` (**поэлементно**).
- ➕ Для скорости есть варианты с префиксами `native_` (`native_sin`, `native_exp`…) и `half_` — быстрее, но менее точные.
- ➕ `rootn(x,y)` ≠ `pow(x,y)`: `rootn` — корень, `pow` — возведение в степень.
- ➕ `fmod` vs `remainder`: `remainder` округляет частное к ближайшему целому (результат может быть отрицательным), `fmod` — усечение к нулю.

---

## 25. Константы (суффикс `_F` = float)

| Константа | Значение |
|-----------|----------|
| `M_E_F` | e |
| `M_LOG2E_F` | log₂e |
| `M_LOG10E_F` | log₁₀e |
| `M_LN2_F` | ln 2 (logₑ2) |
| `M_LN10_F` | ln 10 (logₑ10) |
| `M_PI_F` | π |
| `M_PI_2_F` | π / 2 |
| `M_PI_4_F` | π / 4 |
| `M_1_PI_F` | 1 / π |
| `M_2_PI_F` | 2 / π |
| `M_2_SQRTPI_F` | 2 / √π |
| `M_SQRT2_F` | √2 |
| `M_SQRT1_2_F` | 1 / √2 |

- Суффикс `_F` — вариант для типа `float`. Без суффикса (`M_PI`) — для `double`.
- Логика имён: `M_PI_2` — «π на 2», `M_1_PI` — «1 делить на π», `M_2_PI` — «2 делить на π» (число в имени стоит там, где оно в формуле).

---

## 26. Геометрические функции

| Функция | Смысл |
|---------|-------|
| `float3 cross(float3 p0, float3 p1)` | **векторное произведение** (только для 3 и 4 компонент) |
| `float dot(floatn p0, floatn p1)` | **скалярное произведение** |
| `float distance(floatn p0, floatn p1)` | **расстояние** между точками |
| `float length(floatn p)` | **евклидова норма** (длина вектора) |
| `floatn normalize(floatn p)` | **нормирование** вектора (длина станет 1) |

Связи: `distance(p0, p1) = length(p0 - p1)`; `normalize(p) = p / length(p)`; `length(p) = sqrt(dot(p, p))`.

---

## 27. Функции сравнения

| Функция | Смысл |
|---------|-------|
| `intn isequal(floatn x, floatn y)` | сравнение «равно»; возвращает вектор: **0 — false, −1 — true** |
| `intn isnotequal(...)` | не равно |
| `intn isgreater(...)` | больше |
| `intn isgreaterequal(...)` | больше или равно |
| `intn isless(...)` | меньше |
| `intn islessequal(...)` | меньше или равно |
| `int any(igentypen x)` | возвращает ненулевое, если **старший бит хотя бы у одного** элемента вектора равен 1 |
| `int all(igentypen x)` | возвращает ненулевое, если **старший бит у всех** элементов равен 1 |

**Ключевые моменты:**
- **Для векторов** результат сравнения: `true = −1` (все биты единицы, то есть старший бит = 1), `false = 0`. **Для скаляров** `true = 1`.
- Именно поэтому `any`/`all` проверяют **старший бит** (знаковый): у `−1` он равен 1, у `0` — нулю. Так `any(isequal(a, b))` означает «есть хотя бы одно совпадение», `all(...)` — «совпадают все компоненты».
- Результат сравнения `floatn` — вектор `intn` того же размера.

> ⚠ Ловушка: на слайде написано «если элемент = 1 → это false». Это сформулировано **неточно/наоборот**: если старший бит элемента равен 1 (отрицательное число, например −1), то элемент считается **истинным** (true). Для `any`/`all` «1 в старшем бите» = «true».

---

# ЧАСТЬ 3. Шпаргалка и вопросы для самопроверки

## 28. Быстрая шпаргалка

**Иерархия:** NDRange → Work Group → Work Item  (Grid → Block → Thread).

**Память:** Global (всем) · Constant (всем, только чтение) · Local (группе) · Private (work-item).

**Quick-map CUDA → OpenCL:**

```
__global__      → __kernel
__device__ func → (ничего)
__shared__      → __local
__constant__    → __constant
__device__ var  → __global
__syncthreads() → barrier(CLK_LOCAL_MEM_FENCE)
threadIdx.x     → get_local_id(0)
blockIdx.x      → get_group_id(0)
blockDim.x      → get_local_size(0)
gridDim.x       → get_num_groups(0)
global index    → get_global_id(0)
total threads   → get_global_size(0)
```

**Порядок вызовов хоста:**
`clGetPlatformIDs → clGetDeviceIDs → clCreateContext → clCreateCommandQueue → clCreateProgramWithSource → clBuildProgram → clCreateKernel → clCreateBuffer → clSetKernelArg → clEnqueueNDRangeKernel → clFinish → clEnqueueReadBuffer`

**Чего нет в OpenCL C:** заголовков C99, указателей на функции, рекурсии, VLA, битовых полей.

---

## 29. Вопросы для самопроверки (с ответами)

1. **Что такое платформа в OpenCL?** — Драйвер/реализация OpenCL от производителя; список получают `clGetPlatformIDs`.
2. **Что возвращает `clGetPlatformIDs` в `numPlatforms`?** — Количество доступных драйверов OpenCL (можно передать NULL).
3. **Какие типы устройств бывают?** — CPU, GPU, ACCELERATOR, DEFAULT, ALL.
4. **Что такое контекст?** — Окружение, объединяющее устройства, память, программы, ядра.
5. **Зачем нужна очередь команд?** — Через неё хост ставит устройству команды (запуск ядра, копирование). Одна очередь — одно устройство.
6. **Что задают `properties` очереди?** — Возможность выполнения не по порядку (out-of-order) и разрешение профилирования.
7. **Когда компилируется ядро?** — Во время работы хост-программы (`clCreateProgramWithSource` + `clBuildProgram`).
8. **Зачем callback в `clBuildProgram`?** — Уведомление о завершении сборки (делает её асинхронной).
9. **Что делает `clCreateKernel`?** — Создаёт объект ядра по имени `__kernel`-функции из собранной программы.
10. **Чем `CL_MEM_USE_HOST_PTR` отличается от `CL_MEM_COPY_HOST_PTR`?** — USE: буфер использует память хоста напрямую; COPY: данные копируются в буфер при создании.
11. **Что означает `arg_index` в `clSetKernelArg`?** — Порядковый номер параметра ядра (с 0).
12. **Что означает `work_dim`?** — Размерность пространства work-item (1, 2 или 3).
13. **Чем `clFinish` отличается от `clWaitForEvents`?** — `clFinish` ждёт всю очередь, `clWaitForEvents` — конкретные события.
14. **Что значит `CL_TRUE` в `clEnqueueReadBuffer`?** — Блокирующее чтение: возврат только после копирования.
15. **В каких единицах время профилирования?** — Наносекунды (поэтому `/1e9` → секунды).
16. **Что нужно для профилирования?** — Очередь с `CL_QUEUE_PROFILING_ENABLE` + событие от команды.
17. **Что такое Work group / Work item / NDRange?** — Аналоги Block / Thread / Grid.
18. **Чему в CUDA соответствует Local memory OpenCL?** — Shared memory. А Private memory OpenCL = Local memory CUDA.
19. **Аналог `__syncthreads()`?** — `barrier()`.
20. **Как вычислить глобальный индекс?** — `get_global_id(0)` = `get_group_id(0) * get_local_size(0) + get_local_id(0)`.
21. **Что запрещено в OpenCL C по сравнению с C99?** — Заголовки C99, указатели на функции, рекурсия, VLA, битовые поля.
22. **Какие размеры у векторных типов?** — 2, 4, 8, 16.
23. **Что вернёт `f.odd` для `float4 f = (1,2,3,4)`?** — `(2, 4)`, т.е. `f.yw`. А `f.even` → `(1, 3)`.
24. **Как обратиться к 10-й компоненте `float16`?** — `x.sa` или `x.sA`.
25. **Что вернёт `isequal` для векторов?** — Вектор `intn`: 0 — false, −1 — true.
26. **Как вычисляется `clamp(x, a, b)`?** — `min(max(x, a), b)`.
27. **Что делает `mad(a,b,c)`?** — `a*b + c`.
28. **Чем `step` отличается от `smoothstep`?** — `step` — резкий скачок (Хевисайд), `smoothstep` — плавный переход между `edge0` и `edge1`.
29. **Что делает `rootn(x,y)`?** — `x^(1/y)`.
30. **Чему равна `M_2_SQRTPI_F`?** — 2/√π.

---

## 30. Список всех «ловушек» с исправлениями (повторить перед тестом)

| Где | На слайде | Как правильно |
|-----|-----------|---------------|
| `CL_DEVICE_TYPE_DEFAULT` | «по молчанию» | «по умолчанию» |
| `CL_MEM_READ_ONLY` | «только запись» | **только чтение** |
| Создание очереди | `properties = 0`, но дальше профилирование | для профилирования нужен `CL_QUEUE_PROFILING_ENABLE` |
| Индексы `float16` | пропущены `d, D` | `0–9, a–f, A–F` (всего 16) |
| Пример `float16 x; f.sa…` | обращение к `f` | должно быть `x.sa`, `x.sF` |
| `any`/`all` | «если элемент = 1 → false» | старший бит = 1 → **true** (−1) |
| Kernel `sum` | лишние `uint i, j` и `int i` | просто затенение, на результат не влияет |
| `local` в CUDA vs OpenCL | — | CUDA Local = OpenCL **Private**; CUDA Shared = OpenCL **Local** |

---

# ЧАСТЬ 4. НЮАНСЫ, ПОДВОДНЫЕ КАМНИ И ОПТИМИЗАЦИЯ (для каверзных вопросов)

> Здесь собрано то, что **не написано на слайдах**, но на строгом тесте спрашивают именно это: «а что будет, если…», «почему так работает», «как ускорить».
> Помечено: **[1.x]** — поведение OpenCL 1.x (как в презентации), **[2.0+]** — что изменилось позже.

---

## 31. Нюансы хост-API: что происходит «под капотом»

### 31.1. Общие правила

- **Почти все функции возвращают `cl_int`-код ошибки**, а функции, создающие объекты (`clCreateContext`, `clCreateBuffer`, `clCreateKernel`, `clCreateProgramWithSource`, `clCreateCommandQueue`), возвращают **сам объект**, а код ошибки кладут в `errcode_ret`. Если там `NULL` (как на слайдах) — **ошибка не будет видна**, вернётся `NULL`-объект, и программа упадёт позже в непонятном месте.
- `CL_SUCCESS == 0`. Все ошибки — отрицательные числа (`CL_DEVICE_NOT_FOUND = -1`, `CL_BUILD_PROGRAM_FAILURE = -11`, `CL_INVALID_VALUE = -30`, `CL_INVALID_WORK_GROUP_SIZE = -54`, и т.д.).
- Все объекты OpenCL (`cl_context`, `cl_mem`, `cl_kernel`, `cl_program`, `cl_command_queue`, `cl_event`) — **непрозрачные дескрипторы** с **подсчётом ссылок**. Освобождение: `clReleaseKernel`, `clReleaseProgram`, `clReleaseMemObject`, `clReleaseCommandQueue`, `clReleaseContext`, `clReleaseEvent`. Увеличить счётчик: `clRetain...`. Объект реально удаляется, когда счётчик = 0 **и** все команды, которые его используют, завершены. На слайдах освобождения нет — в реальной программе это утечка.
- `cl_mem` — **не указатель** на память устройства: нельзя разыменовывать на хосте, нельзя делать арифметику указателей (в отличие от `cudaMalloc`-указателя). Смещение задают параметром `offset` в функциях копирования.
- **libOpenCL.so — это ICD-загрузчик** (Installable Client Driver): он сам не выполняет вычисления, а находит драйверы производителей (в Linux — файлы `/etc/OpenCL/vendors/*.icd`) и перенаправляет вызовы. **Каждый драйвер = одна платформа.**

### 31.2. `clGetPlatformIDs`

- Типичный приём **«два вызова»**: `clGetPlatformIDs(0, NULL, &n)` → узнали `n` → выделили массив → `clGetPlatformIDs(n, arr, NULL)`.
- Ошибки: `CL_INVALID_VALUE`, если `numEntries == 0`, но `platforms != NULL`; либо если `platforms == NULL` **и** `numPlatforms == NULL` одновременно (вызов бессмысленен).
- Если вернулось больше платформ, чем `numEntries`, — записываются только первые `numEntries`, а `numPlatforms` содержит **реальное** количество.
- В слайде массив на 10: если платформ больше 10, лишние просто не получим.
- Информацию о платформе (имя, версия, производитель) получают `clGetPlatformInfo`.

### 31.3. `clGetDeviceIDs` — тонкости кода со слайда

```c
clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &device, &num_gpgpu);
if (num_gpgpu == 0) { clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_CPU, 1, &device, NULL); }
```

- Если устройств нужного типа **нет**, функция возвращает ошибку **`CL_DEVICE_NOT_FOUND`**, а `num_gpgpu` при ошибке **может остаться неинициализированным** (мусор!). Надёжнее проверять **код возврата**, а не только счётчик (а счётчик заранее обнулять).
- Код предполагает, что GPU и CPU находятся **на одной платформе `platform[0]`**. Это не гарантировано: у NVIDIA платформа есть только GPU, у Intel/POCL — CPU. Правильно перебирать все платформы.
- Если не нашлось ни GPU, ни CPU, `device` остаётся неинициализированным — дальше будет сбой.
- `nNumEntries = 1` — запросили **максимум одно** устройство (первое найденное). Для всех устройств нужен массив и `CL_DEVICE_TYPE_ALL`.
- Типы — это **битовая маска**: можно комбинировать `CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU`.
- Характеристики устройства: `clGetDeviceInfo` — например `CL_DEVICE_MAX_COMPUTE_UNITS`, `CL_DEVICE_MAX_WORK_GROUP_SIZE`, `CL_DEVICE_MAX_WORK_ITEM_SIZES`, `CL_DEVICE_LOCAL_MEM_SIZE`, `CL_DEVICE_GLOBAL_MEM_SIZE`, `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE`, `CL_DEVICE_EXTENSIONS`.

### 31.4. Контекст

- Контекст **не привязан к одному устройству**: в него можно включить несколько устройств **одной платформы** (из разных платформ — нельзя).
- Буфер (`cl_mem`) принадлежит **контексту**, а не устройству: рантайм сам переносит данные на то устройство, где запускается ядро.
- **Callback `pfnNotify`** вызывается **асинхронно**, возможно из другого потока драйвера; код внутри должен быть **потокобезопасным** и не вызывать блокирующих OpenCL-функций. Он сообщает только об **ошибках, возникших уже в процессе работы** контекста (не о неверных параметрах вызова — те возвращаются кодом).
- `properties = NULL` → платформа выбирается реализацией по умолчанию; обычно передают `{CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0}` — массив, **заканчивающийся нулём**.

### 31.5. Очередь команд

- **In-order (по умолчанию):** команды стартуют в порядке постановки, и следующая начинается после завершения предыдущей (с т.з. видимых результатов). Поэтому после `clEnqueueNDRangeKernel` блокирующий `clEnqueueReadBuffer` **сам дождётся** конца ядра.
- **Out-of-order** (`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`): порядок не гарантируется; зависимости надо задавать явно через **события** (`event_wait_list`) или барьеры (`clEnqueueBarrier`/`clEnqueueMarker`). Не все драйверы реально поддерживают.
- Очередь **асинхронна**: функции `clEnqueue*` лишь *ставят* команду и возвращаются. Реальное выполнение может начаться позже.
- `clFlush(queue)` — **отправляет** накопленные команды на устройство, но **не ждёт**. `clFinish(queue)` — flush + **ожидание** завершения всех команд. Без `clFlush`/`clFinish` команда может вообще не начать выполняться (драйвер копит их).
- **Одна очередь — одно устройство.** Для нескольких устройств — несколько очередей. Для **перекрытия** копирования и вычислений нужны **разные очереди** (или out-of-order).
- `clCreateCommandQueue` **[2.0+]** — deprecated, заменена `clCreateCommandQueueWithProperties`.

### 31.6. Программа и сборка

- **Компиляция «на лету» (online/JIT)** при каждом запуске хост-программы. Это существенное время (десятки–сотни мс) — **не включается** в замер ядра, но влияет на старт программы. Чтобы не компилировать каждый раз: получить бинарник (`clGetProgramInfo` с `CL_PROGRAM_BINARIES`) и загружать через `clCreateProgramWithBinary`. Бинарник **привязан к устройству/драйверу**.
- **Частая ошибка с исходником:** если `lengths == NULL`, строка **должна заканчиваться `'\0'`**. При чтении из файла нужно вручную дописать нулевой символ — иначе в код попадает мусор и сборка падает.
- `clBuildProgram` вернул ошибку → **причину** смотреть в логе: `clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ...)`. Сама функция текст ошибки компилятора не возвращает.
- **Опции сборки (`options`):**
  - `-D NAME=VALUE` — макроопределение (удобно задавать размеры при компиляции);
  - `-I путь` — каталог для `#include` в `.cl`-файле;
  - `-cl-fast-relaxed-math` — быстрая математика (меньше точность, возможна потеря корректной обработки NaN/Inf);
  - `-cl-mad-enable` — разрешить замену `a*b+c` на `mad`;
  - `-cl-single-precision-constant` — трактовать `double`-константы как `float`;
  - `-cl-opt-disable` — отключить оптимизации (для отладки);
  - `-w` (без предупреждений), `-Werror` (предупреждения как ошибки).
- Нельзя пересобирать программу (`clBuildProgram`), пока к ней привязаны созданные kernel-объекты → `CL_INVALID_OPERATION`.
- **Имя ядра в `clCreateKernel` — строка, регистр важен.** Опечатка → `CL_INVALID_KERNEL_NAME`.

### 31.7. Буферы (`clCreateBuffer`) — тонкости флагов

- **Если флаги доступа не указаны — по умолчанию `CL_MEM_READ_WRITE`.**
- **Допустимые комбинации:**
  - `USE_HOST_PTR` и `ALLOC_HOST_PTR` **несовместимы**;
  - `USE_HOST_PTR` и `COPY_HOST_PTR` **несовместимы**;
  - `ALLOC_HOST_PTR | COPY_HOST_PTR` — можно;
  - `USE_HOST_PTR` / `COPY_HOST_PTR` **требуют ненулевого `host_ptr`**, иначе `CL_INVALID_HOST_PTR`. И наоборот, если этих флагов нет, `host_ptr` должен быть `NULL`.
  - `READ_WRITE`, `WRITE_ONLY`, `READ_ONLY` — взаимоисключающие.
- `CL_MEM_READ_ONLY` / `WRITE_ONLY` ограничивают доступ **со стороны ядра**, а не хоста: хост через `clEnqueueRead/WriteBuffer` всё равно может читать/писать. Попытка ядра писать в `READ_ONLY` буфер — **неопределённое поведение**.
- **Выделение может быть ленивым:** `clCreateBuffer` не обязан сразу выделять память на устройстве; реально память выделяется/копируется при **первом использовании** (первом запуске ядра). Поэтому **первый запуск ядра медленнее** остальных, и его не стоит включать в замер.
- `COPY_HOST_PTR` — копирование происходит при создании; это **разовая** копия, дальнейшие изменения массива на хосте на устройство не попадают.
- `USE_HOST_PTR`: на **CPU-устройстве** это может дать **нулевое копирование** (устройство работает прямо в памяти хоста); на GPU драйвер всё равно часто копирует. Содержимое буфера, пока он используется ядром, на хосте трогать нельзя (результат не определён) — только через `clEnqueueMapBuffer`.
- `ALLOC_HOST_PTR` — **закреплённая (pinned) память** хоста: передачи по PCIe быстрее, но ресурс ограничен.
- Размер `size` — **в байтах**. Классическая ошибка — забыть `* sizeof(float)`.
- Превышение лимитов: `CL_DEVICE_MAX_MEM_ALLOC_SIZE` (максимальный размер одного буфера, часто ¼ памяти устройства) → `CL_INVALID_BUFFER_SIZE`/`CL_MEM_OBJECT_ALLOCATION_FAILURE`.

### 31.8. `clSetKernelArg` — подводные камни

- **Типы не проверяются и не преобразуются!** Размер `arg_size` должен **точно совпадать** с типом параметра в ядре. Если параметр `float`, а вы передали `int n` — байты просто будут **интерпретированы как float** (мусор). Используйте типы `cl_int`, `cl_float`, `cl_uint` — размер которых гарантирован.
- Для буфера: `arg_size = sizeof(cl_mem)`, `arg_value = &buf` (**указатель на переменную `cl_mem`**, а не сам хостовый массив).
- **Значение копируется в момент вызова** (для скаляров) — переменную можно сразу менять/переиспользовать.
- **Аргументы «запоминаются»** в kernel-объекте и сохраняются между запусками: если не менялись, повторно вызывать `clSetKernelArg` не нужно.
- Если установлены не все аргументы — при `clEnqueueNDRangeKernel` будет `CL_INVALID_KERNEL_ARGS`. Неверный размер — `CL_INVALID_ARG_SIZE`. Неверный индекс — `CL_INVALID_ARG_INDEX`.
- **Для `__local` параметра** передают **размер и `NULL`**: `clSetKernelArg(kernel, 3, local_size * sizeof(int), NULL);` — так хост задаёт размер локального массива.
- Объект kernel **не потокобезопасен** для `clSetKernelArg` (нельзя менять аргументы одного kernel из разных потоков одновременно).
- В ядре **нельзя** принимать аргументами: `bool`, `size_t`, `ptrdiff_t`, указатели на указатели (в 1.x) — не имеют определённого размера/представления на хосте и устройстве.

### 31.9. `clEnqueueNDRangeKernel` — правила размеров (очень частый вопрос!)

- `global_work_size[i]` — **общее число work-item** по измерению `i` (в CUDA — число *блоков* × размер блока; тут сразу итоговое число потоков).
- **[1.x]** `global_work_size[i]` **должен нацело делиться** на `local_work_size[i]`, иначе `CL_INVALID_WORK_GROUP_SIZE`. Решение: округлить глобальный размер **вверх** до кратного локальному и добавить в ядро проверку `if (i < n)`:

```c
size_t local  = 128;
size_t global = ((n + local - 1) / local) * local;   // округление вверх
```

- **[2.0+]** разрешены work-group **неравного размера** (если не указать `-cl-uniform-work-group-size`).
- `local_work_size = NULL` — размер группы выбирает **реализация** (как в слайде). Он может оказаться неоптимальным (даже 1). Если задан, ограничения такие:
  - произведение размеров ≤ `CL_DEVICE_MAX_WORK_GROUP_SIZE` (на GPU обычно 256/512/1024);
  - каждое измерение ≤ соответствующему элементу `CL_DEVICE_MAX_WORK_ITEM_SIZES`;
  - ещё и ограничение **конкретного ядра** `CL_KERNEL_WORK_GROUP_SIZE` (через `clGetKernelWorkGroupInfo`) — оно уменьшается, если ядро использует много регистров/локальной памяти.
- `work_dim` ∈ {1, 2, 3}; массивы размеров должны содержать ровно `work_dim` элементов.
- `global_work_offset`: **[1.0]** обязан быть `NULL`; **[1.1+]** допускается. Смещение учитывается в `get_global_id()`, но **не** влияет на `get_local_id()`, `get_group_id()`.
- Если `num_events_in_wait_list == 0`, то `event_wait_list` должен быть `NULL` (и наоборот: ненулевое число ↔ непустой список), иначе `CL_INVALID_EVENT_WAIT_LIST`.
- **Слайд `gridSize = 10`:** запускается всего 10 work-item. Для функциональности (учебный пример) достаточно — благодаря циклу с шагом `sz` ядро обработает массив любой длины. Но **для производительности это плохо**: GPU нужны тысячи work-item для заполнения устройства. Это важный нюанс «почему такой код не быстрее CPU».
- Запуск **не блокирует хост**. `event` нужно потом освободить `clReleaseEvent`.

### 31.10. Чтение/запись буферов

- **Блокирующее** (`CL_TRUE`) — возврат после завершения копирования; можно сразу использовать `ptr`. **Неблокирующее** (`CL_FALSE`) — возврат сразу; **`ptr` должен оставаться валидным** до завершения (проверять через event), иначе читаем недозаписанные данные или портим память.
- Чтение — это **команда в очереди**: оно ждёт всех предыдущих команд (в in-order очереди) — отсюда «дождёмся ядра».
- `offset + cb` не должны выходить за размер буфера → `CL_INVALID_VALUE`.
- Парная функция: `clEnqueueWriteBuffer` (хост → устройство). Копирование внутри устройства: `clEnqueueCopyBuffer`. Отображение в память хоста без копирования (где возможно): `clEnqueueMapBuffer` / `clEnqueueUnmapMemObject`.
- Пример типовой ошибки: результат `res` на устройстве **не обнулён** до запуска ядра `sum`, где используется `atomic_add` — получится «мусор + сумма». Надо создавать буфер с `COPY_HOST_PTR` из обнулённого значения или записать ноль через `clEnqueueWriteBuffer`.

### 31.11. События и профилирование — нюансы

- Профилировать можно **только** если очередь создана с `CL_QUEUE_PROFILING_ENABLE`, иначе `CL_PROFILING_INFO_NOT_AVAILABLE`.
- Информация доступна **только после завершения команды** (статус `CL_COMPLETE`) — поэтому перед `clGetEventProfilingInfo` вызывается `clWaitForEvents`.
- Метки: `CL_PROFILING_COMMAND_QUEUED` → `SUBMIT` → `START` → `END`. Разность `END − START` — **чистое время выполнения на устройстве**. Разность `START − QUEUED` — накладные расходы очереди/запуска.
- Время — 64-бит **наносекунды** (`cl_ulong`); разрешение таймера — `CL_DEVICE_PROFILING_TIMER_RESOLUTION`. Деление на `1e9` даёт секунды.
- Замер через `clock()`/`gettimeofday` вокруг `clEnqueueNDRangeKernel` **без** `clFinish` измеряет лишь время постановки в очередь (микросекунды!), а не вычисление.
- **Правильная методика замера:** (1) «прогревочный» запуск (JIT, ленивое выделение), (2) несколько запусков и усреднение/минимум, (3) отдельно считать время **передачи** данных и **вычислений**, (4) сравнивать с CPU честно (учитывая передачи).
- Для ускорения важно знать: событие `cl_event` **нужно освобождать**.

### 31.12. Макросы проверки ошибок со слайда — нюансы C-препроцессора

- `#func` — **оператор строкизации**: аргумент макроса превращается в строку (покажет текст команды в сообщении).
- Макросы жёстко используют переменную **`errcode`** (она должна быть объявлена в области видимости).
- `checkErrorEx(command)` раскрывается в **два оператора** (`command; if (...) {...}`), поэтому использовать его в `if` без фигурных скобок **нельзя** (ловушка «висячего else»). Безопаснее оборачивать в `do { ... } while(0)`.
- Макрос `checkError` в вызовах с `errcode = clXxx(...)` проверяет результат, **присвоенный в `errcode`**; если вызвать `checkErrorEx(clXxx(...))` без присваивания — проверяться будет старое значение `errcode`!
- `exit(1)` не освобождает ресурсы OpenCL — в учебном коде допустимо.

---

## 32. Нюансы OpenCL C (язык ядер)

### 32.1. Функции индексации — детали

- Возвращаемый тип `get_global_id`, `get_local_id`, `get_global_size` и т.д. — **`size_t`** (на 64-битных устройствах 64 бита). Присваивание в `uint`/`int` — неявное сужение (допустимо для небольших значений; на слайде `uint id = get_global_id(0)`).
- **Если `dim ≥ work_dim`:** `get_global_id/local_id/group_id` возвращают **0**, а `get_global_size/local_size/num_groups` — **1**. То есть при одномерном запуске `get_global_id(1) == 0`, `get_global_size(1) == 1`.
- `get_work_dim()` — число измерений запуска; `get_global_offset(d)` — смещение.
- Связи (без offset): `get_global_id(d) = get_group_id(d) * get_local_size(d) + get_local_id(d)`; `get_num_groups(d) = get_global_size(d) / get_local_size(d)`.
- Если при запуске `local_work_size = NULL`, то `get_local_size()` вернёт то, что выбрала реализация (заранее неизвестно).
- Линейный индекс в 2D: `i = get_global_id(1) * get_global_size(0) + get_global_id(0)` — **ось 0 «быстрая»** (как x в CUDA). Для эффективного доступа к памяти соседние work-item по оси 0 должны обращаться к соседним адресам.

### 32.2. Адресные пространства — правила и ограничения

| Квалификатор | Где живёт / видимость | Нюансы |
|--------------|-----------------------|--------|
| `__global` | глобальная память устройства, видна всем | самая медленная (сотни тактов задержки), нужен coalescing |
| `__constant` | константная память, только чтение | ограничена: `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` (минимум **64 КБ**); кэшируется; быстро, когда **все work-item читают один адрес** (broadcast) |
| `__local` | на work-group, общая | ограничена: `CL_DEVICE_LOCAL_MEM_SIZE` (минимум **32 КБ**); **не инициализируется** (в ней мусор); значения видны другим work-item **только после барьера** |
| `__private` | у каждого work-item (регистры) | по умолчанию для переменных внутри функций; при нехватке регистров «проливается» (spill) в медленную память |

- **Аргументы-указатели `__kernel`-функции** обязаны указывать на `__global`, `__constant` или `__local`. Указатель на `__private` в аргументе ядра — нельзя.
- **В OpenCL 1.x нельзя преобразовывать указатели между адресными пространствами** (`__global int*` ↔ `__local int*`). Присваивание/приведение — ошибка. Функции-помощники приходится писать под конкретное пространство. **[2.0+]** появилось «generic»-пространство.
- **Переменные уровня программы (вне функций)** в OpenCL 1.x могут быть **только `__constant`**. Поэтому соответствие со слайда «`__device__` variable → `__global` variable» работает для **параметров/буферов**, а не для глобальных переменных C-стиля **[1.x]**. **[2.0+]** разрешает program-scope `__global`.
- `__local`-переменные внутри ядра объявляют **внутри `__kernel`-функции** (статический размер): `__local float tile[16][16];`. Инициализатора у них быть не может.
- **Нет динамического выделения памяти** (`malloc`/`free` отсутствуют) — размеры известны заранее либо задаются с хоста.
- `__global` указатель в разных `clSetKernelArg` могут указывать на один и тот же буфер — **алиасинг**; чтобы помочь компилятору, используют `restrict` и `const`.

### 32.3. Что нельзя в OpenCL C (дополнение к слайду)

Слайд: нет стандартных заголовков C99, указателей на функции, рекурсии, VLA, битовых полей. Дополнительно:

- нет `malloc`/`free`, `stdio.h` (`printf` появился в **OpenCL 1.2**), `stdlib.h`;
- нет `long double`; `double` — **только** с расширением `cl_khr_fp64`:

```c
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
```

- **Литерал `1.0` — это `double`**, а `1.0f` — `float`. Поэтому в ядрах везде пишут `f` (как на слайдах `1.0f`): иначе без поддержки double — ошибка, с поддержкой — **резкое падение скорости** (особенно на игровых GPU) и неявные преобразования;
- нет рекурсии → алгоритмы с рекурсией переписывают итеративно (стек);
- ядро всегда возвращает **`void`**; результат передаётся только через буферы;
- в **1.x** ядро не может запускать другие ядра (динамический параллелизм — только 2.0+).

### 32.4. Типы данных — нюансы

- **Размеры скаляров фиксированы стандартом:** `char` 8, `short` 16, `int` 32, **`long` 64 бита** (даже там, где в обычном C на Windows `long` — 32 бита!), `float` 32, `double` 64. Поэтому на хосте нужно использовать **`cl_int`, `cl_long`, `cl_float`**, а не `int`/`long`.
- **`float3` занимает столько же места, сколько `float4`:** `sizeof(float3) == 16`, выравнивание 16 байт. Массив `float3` в буфере **не плотный**! Для плотного чтения: `vload3`/`vstore3`. Это классическая ловушка при передаче массива координат (x,y,z).
- Выравнивание вектора = его размеру (`float4` — 16 байт, `float8` — 32, `float16` — 64) → векторные обращения к памяти должны быть выровнены.
- **Векторные литералы** пишутся как `(float4)(1.0f, 2.0f, 3.0f, 4.0f)` — это **не приведение типа C**, а особый синтаксис. `(float4)(1.0f)` — **размножает** скаляр во все компоненты.
- **Скаляр в операции с вектором автоматически «расширяется»:** `float4 v; v * 2.0f` — умножает каждую компоненту.
- **Неявное преобразование между разными векторными типами запрещено** (`float4 → int4` и `float4 → float2` — ошибка). Нужны:
  - `convert_int4(v)` (также `convert_int4_sat`, `_rte`, `_rtz` — режимы насыщения/округления) — **преобразование значений**;
  - `as_int4(v)` — **переинтерпретация битов** (как `union`) без изменения битов; размеры должны совпадать.
- **Операции над векторами — поэлементные.** `a + b`, `a * b` — по компонентам (не скалярное/матричное умножение!). Скалярное произведение — только `dot`.
- **Операторы сравнения над векторами** (`a < b`, `a == b`) возвращают вектор знакового целого, где `true = −1`, `false = 0` (для скаляров — `1`/`0`). То же и у функций `isequal`, `isless` и т.д.
- **Тернарный оператор** с векторным условием `c ? a : b` выбирает покомпонентно по **старшему биту** компонент `c`. Для явного выбора есть `select(a, b, c)` и `bitselect(a, b, c)`.
- **Swizzle (перестановки):**
  - при **чтении** можно повторять компоненты (`pos.xxyy`);
  - при **записи** (слева от `=`) **повторы запрещены** (`v.xx = ...` — ошибка);
  - нельзя **смешивать** обозначения `xyzw` и `s0..sF` в одном выражении (`v.xs1` — ошибка);
  - `.x/.y/.z/.w` допустимы только для векторов ≤ 4 компонент; для `float8/float16` — только `.s0..sF`, `.lo/.hi/.odd/.even`;
  - размер результата определяется числом выбранных компонент (`pos.xw` → `float2`), и размеры слева и справа обязаны совпадать.
- `.lo/.hi/.odd/.even` можно применять рекурсивно/цепочкой: `x.hi.lo` (у `float16` → `float4`).

### 32.5. Синхронизация и модель выполнения — «суровые» нюансы

- **Барьер `barrier(flags)` должен быть выполнен *всеми* work-item группы** на одном и том же месте в коде. Если поставить его внутри `if`, куда зайдёт лишь часть потоков, — **зависание или неопределённое поведение**. Внутри цикла число итераций с барьером должно быть **одинаковым** у всех work-item группы.
- Флаги: `CLK_LOCAL_MEM_FENCE` — гарантирует, что записи в `__local` стали видимы остальным; `CLK_GLOBAL_MEM_FENCE` — то же для `__global` **внутри группы**. Сам барьер синхронизирует выполнение (все дошли до точки) **независимо** от флага; флаги отвечают за **видимость памяти**.
- **Синхронизации между work-group внутри одного ядра нет.** Если нужен глобальный барьер — **завершить ядро и запустить следующее** (последовательные запуски в in-order очереди выполняются строго по порядку).
- **Порядок выполнения work-group не определён**, и **не все группы выполняются одновременно** (устройство берёт на выполнение столько групп, сколько помещается). Поэтому нельзя «ждать» (spin-wait) результата другой группы — получите взаимоблокировку.
- **Гонки данных (race condition):** если несколько work-item пишут в один адрес `__global` без атомарных операций — результат **неопределён**. В ядре `sum` именно поэтому используется `atomic_add`.
- Видимость записей **между work-item разных групп** в ходе одного запуска не гарантируется (кроме атомарных операций).
- **Warp/wavefront** выполняется в «lock-step» (SIMT): **расхождение ветвлений (divergence)** внутри warp приводит к тому, что выполняются **обе ветки последовательно** с маскированием потоков. Размер: **32 у NVIDIA, 64 (или 32 в новых) у AMD**; на CPU понятия по сути нет.

### 32.6. Атомарные операции — нюансы (ядро `sum` со слайда)

- `atomic_add`, `atomic_sub`, `atomic_inc`, `atomic_dec`, `atomic_min`, `atomic_max`, `atomic_and/or/xor`, `atomic_xchg`, `atomic_cmpxchg` — работают над **32-битными `int`/`uint`** в `__global` и `__local`.
- **[1.0]** они были расширениями (`cl_khr_global_int32_base_atomics`, имя `atom_add`); **[1.1+]** — в ядре языка, имя `atomic_add`.
- **Атомарное сложение `float` в 1.x не предусмотрено** (`atomic_xchg` умеет `float`). Приём: цикл `atomic_cmpxchg` с `as_uint`/`as_float`.
- Атомарная операция **возвращает старое значение**.
- **Атомарные операции на один адрес сериализуются** → в `sum` все work-item делают `atomic_add` в `res[0]` — это узкое место (высокая конкуренция, contention).
- Целочисленная сумма **детерминирована** (сложение ассоциативно и без округлений, но возможно переполнение `int`). Для `float` порядок сложения влияет на результат из-за округлений → результаты могут различаться от запуска к запуску и от CPU к GPU.

---

## 33. Нюансы встроенных функций

### 33.1. Математика

- **`gentype`** — `float`, `floatn` (и `double` при fp64); функции работают **поэлементно**.
- **`sincos(x, &c)`** — возвращает `sin(x)`, а `cos(x)` пишет по указателю; указатель обязан иметь адресное пространство (`__private`/`__global`/`__local`): `float c; float s = sincos(x, &c);`. Быстрее, чем вызывать `sin` и `cos` по отдельности.
- **`atan2(y, x)`** — **первым идёт `y`!** Результат в диапазоне **(−π, π]**, корректно учитывает **квадрант** и случай `x = 0` — в отличие от `atan(y/x)`, который даёт лишь (−π/2, π/2). На слайде «atan2(y,x) = arctg(y/x)» — упрощение.
- **Семейство `...pi`**: `atanpi`, `asinpi`, `acospi`, `atan2pi` возвращают результат **в единицах π** (то есть значение/π); `sinpi`, `cospi`, `tanpi` принимают аргумент в единицах π.
- **`pow(x, y)`**, **`pown(x, int n)`** (целая степень), **`powr(x, y)`** (при `x ≥ 0`), **`rootn(x, int n)`** = `x^(1/n)`. `rootn` — корень **целой** степени.
- **`sqrt(−1)` → NaN**; `log(0) → −inf`; деление на ноль для `float` даёт `inf`, а для целых — **неопределённое поведение**.
- **`expm1(x)`** = `e^x − 1` и **`log1p(x)`** = `ln(1+x)` — для **малых x** гораздо точнее прямой формулы (нет потери точности при вычитании близких чисел).
- **`hypot(x, y)`** — `sqrt(x²+y²)` **без промежуточного переполнения**.
- **`round`** — к ближайшему, **половины — от нуля** (2.5 → 3); **`rint`** — к ближайшему **чётному** (2.5 → 2); **`trunc`** — к нулю; `ceil`/`floor` — вверх/вниз. Все они возвращают **float**, не int.
- **`remainder(x, y)`** = `x − n·y`, где `n` — **ближайшее целое** к `x/y` (при равенстве — чётное) → результат может быть **отрицательным даже при положительных x, y**. **`fmod(x, y)`** — остаток с **усечением** (знак как у `x`). Слайд называет `remainder` просто «остатком от деления» — различие и спрашивают.
- **`fmax/fmin`** — для `float`; при одном аргументе NaN возвращают **другой** аргумент. Для **целых** — функции `max`/`min`.
- **`abs(int)` возвращает `uint`** (беззнаковый тип!), а **`fabs`** — для `float`. Ловушка при смешивании знаковых/беззнаковых.
- **`mad(a,b,c)`** — «примерно `a*b+c`»: точность и способ вычисления **определяются реализацией** (может быть без промежуточного округления, может быть с). **`fma(a,b,c)`** — **строго** одно округление (точное fused multiply-add).
- **`clamp(x, minval, maxval)`** = `min(max(x, minval), maxval)`; результат **не определён**, если `minval > maxval`. Для векторов `minval/maxval` могут быть **скалярами**.
- **`step(edge, x)`** — **сначала `edge`, потом `x`**: возвращает **0.0, если `x < edge`, иначе 1.0** (включая `x == edge` → 1). Это функция Хевисайда.
- **`smoothstep(e0, e1, x)`** — `t = clamp((x−e0)/(e1−e0), 0, 1)`, результат `t²(3 − 2t)`; равен 0 при `x ≤ e0`, 1 при `x ≥ e1`; **не определена при `e0 ≥ e1`**.
- **`sign(x)`** → `1.0`, `0.0` (±0), `−1.0`; **для NaN возвращает 0.0**.
- **`degrees(rad)`, `radians(deg)`** — перевод единиц (названия аргументов на слайде перепутаны местами по записи, смысл: `degrees` принимает радианы, `radians` — градусы).
- **Скорость vs точность:** варианты **`native_sin`, `native_cos`, `native_exp`, `native_log`, `native_sqrt`, `native_divide`, `native_recip`** — **быстрее, но с неопределённой точностью и ограниченной областью определения**; **`half_*`** — промежуточные. Опция `-cl-fast-relaxed-math` переключает на «быстрые» по возможности.
- Константы **`M_PI_F`** и пр. — `float`; **`M_PI`** — `double` (требует fp64).

### 33.2. Геометрические функции

- **Определены только для `float`, `float2`, `float3`, `float4`** (и `double`-аналогов) — **для `float8`/`float16` `dot`, `length`, `normalize`, `distance` не работают**, несмотря на запись `floatn` на слайде.
- **`cross`** — только для **`float3`/`float4`** (у `float4` четвёртая компонента результата равна 0).
- `distance(p0, p1) = length(p0 − p1)`, `length(p) = sqrt(dot(p,p))`, `normalize(p) = p / length(p)`. Нулевой вектор нормировать нельзя (деление на 0) — проверять заранее.
- Быстрые версии: `fast_length`, `fast_distance`, `fast_normalize` — быстрее, менее точны.
- **`dot` возвращает скаляр**, а `*` над векторами — покомпонентное произведение.

### 33.3. Функции сравнения

- **Тип результата:** для скаляра — `int` (1/0); для вектора `floatn` — **`intn`** (−1/0); для `doublen` — `longn`.
- **Сравнения с NaN:** `isequal(NaN, x)` → **false**, `isnotequal(NaN, x)` → **true**. Для проверок: `isnan`, `isinf`, `isfinite`, `isnormal`, `isordered`, `isunordered`, `signbit`.
- **`any(x)` / `all(x)`** принимают **знаковые целые** (`char/short/int/long` и их векторы), возвращают `int` **1 или 0**:
  - `any` → 1, если **старший (знаковый) бит хотя бы у одной** компоненты = 1;
  - `all` → 1, если старший бит **у всех** компонент = 1.
  Поэтому `any(isequal(a,b))` = «есть совпадающая компонента», `all(isequal(a,b))` = «векторы равны покомпонентно». Для `scalar`: `any(−5)` = 1, `any(5)` = 0.
- Различие: оператор `a == b` над векторами тоже поэлементный (результат −1/0), а не «равны ли векторы целиком»; целиком сравнивают `all(a == b)`.

---

## 34. Оптимизация производительности

### 34.1. Главные принципы (по важности)

1. **Минимизировать передачи хост ↔ устройство.** Шина PCIe в десятки раз медленнее памяти GPU. Данные нужно загружать **один раз**, цепочку ядер выполнять на устройстве, результат забирать **один раз**. Для небольших задач передача может «съесть» весь выигрыш (ускорение может быть **<1**).
2. **Достаточное число work-item.** Для загрузки GPU нужны **тысячи–миллионы** work-item (нужно скрывать задержки памяти переключением между warp). `gridSize = 10` на GPU — демонстрационный, а не быстрый.
3. **Эффективный доступ к глобальной памяти (coalescing).** Соседние work-item одного warp/wavefront должны обращаться к **соседним** адресам: тогда обращения объединяются в минимальное число транзакций.
4. **Использовать локальную память** для данных, которые переиспользуются в группе (тайлинг, редукции) — она на порядки быстрее глобальной.
5. **Не допускать расхождения ветвлений** внутри warp.
6. **Высокая арифметическая интенсивность** (много операций на байт): иначе ядро упирается в пропускную способность памяти (memory-bound).
7. **Заполняемость (occupancy):** число одновременно живых групп ограничено **регистрами** и **локальной памятью**, приходящимися на группу. Слишком «толстое» ядро → мало групп → хуже скрываются задержки.

### 34.2. Coalescing на примере ядра `sum`

```c
for (int i = id; i < n; i += sz)  s += x[i];   // как на слайде
```

На каждой итерации work-item с соседними `id` читают **соседние** элементы `x[i]` → обращения **коалесцированы** (хорошо). Плохой вариант — когда каждый work-item обрабатывает **свой непрерывный кусок** массива (`i = id*chunk … (id+1)*chunk`): тогда на одной итерации соседние work-item читают адреса, отстоящие на `chunk` → **множество транзакций**. На **CPU** всё наоборот: там выгоднее непрерывный кусок на поток (кэш).

Паттерн «цикл с шагом = общему числу потоков» (**grid-stride loop**) также делает ядро **независимым от размера массива** и от числа запущенных work-item.

### 34.3. Редукция: как улучшить ядро `sum`

Проблема слайда: **каждый** work-item вызывает `atomic_add` в одну ячейку → атомарные операции **сериализуются**. Улучшение — сначала свернуть в **локальной памяти**, а атомарную операцию выполнять **один раз на группу**:

```c
__kernel void sum(__global const int *x, int n, __global int *res, __local int *buf)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int sz  = get_global_size(0);

    int s = 0;
    for (int i = gid; i < n; i += sz) s += x[i];   // частичная сумма в регистре
    buf[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = get_local_size(0) / 2; off > 0; off >>= 1) {
        if (lid < off) buf[lid] += buf[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);   // барьер вне if — его выполняют все!
    }
    if (lid == 0) atomic_add(&res[0], buf[0]);       // одна атомарка на группу
}
```

Нюансы: размер группы должен быть **степенью двойки**; размер `buf` задаётся с хоста: `clSetKernelArg(kernel, 3, local * sizeof(int), NULL)`; барьер **обязан** стоять вне `if`.

### 34.4. Размер work-group

- Выбирать **кратным размеру warp/wavefront** (32/64). Подсказка от драйвера: `clGetKernelWorkGroupInfo(..., CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, ...)` **[1.1+]**.
- Не больше `CL_KERNEL_WORK_GROUP_SIZE` ядра.
- Типичный диапазон на GPU — **128–256**; слишком малые группы не заполняют устройство и плохо используют локальную память, слишком большие снижают occupancy.
- `NULL` вместо `local_work_size` — удобно, но не всегда оптимально; для критичных ядер подбирать вручную. **В 2D** типично 16×16 (= 256).

### 34.5. Локальная память и bank conflicts

- Локальная память разбита на **банки** (обычно 16 или 32). Если несколько work-item одного warp обращаются к **разным адресам одного банка**, обращения **сериализуются** (bank conflict). Обращение к **одному и тому же** адресу — broadcast, конфликта нет.
- Типичное лечение: **дополнение (padding)** строки массива лишним столбцом, например `__local float tile[16][16+1];` (транспонирование/умножение матриц).
- **Тайлинг (блочное умножение матриц):** каждая группа загружает по блоку `A` и `B` в `__local`, делает `barrier`, считает частичную сумму, снова `barrier` перед загрузкой следующего блока. Сокращает число обращений к глобальной памяти в `TILE` раз. **Два барьера** на итерацию — до вычисления (данные загружены) и после (прежде чем перезаписывать тайл).

### 34.6. Расхождение ветвлений (divergence)

- `if (id % 2)` внутри warp → оба пути выполняются последовательно.
- Лучше: условия, **одинаковые для всего warp** (например, зависящие от `get_group_id`, а не от `get_local_id`); заменять ветвления на арифметику/`select`/`clamp`/`step`.
- Для **векторных** условий `select()` не создаёт ветвления.

### 34.7. Векторизация

- Векторные типы (`float4`) позволяют читать/писать **за одну транзакцию** 16 байт и на **CPU** отображаются на SIMD-инструкции (SSE/AVX).
- **Но на современных скалярных GPU** (NVIDIA, новые AMD) векторизация вычислений **сама по себе не ускоряет** — выигрыш только от широких обращений к памяти. Раньше (AMD VLIW) `float4` давал большой выигрыш.
- Помнить про **выравнивание** и про особенность `float3` (занимает 16 байт).

### 34.8. Выбор типов и математики

- Использовать **`float`**, не `double` (на GPU `double` в 2–32 раза медленнее); константы с суффиксом `f`.
- Заменять деление на умножение на обратное, где допустимо; **целочисленное деление и `%` — дорогие**; степени двойки заменять сдвигами/масками.
- `native_*`, `mad`, `fma`, `-cl-fast-relaxed-math` — ускоряют ценой точности.
- Использовать **`const`** и **`restrict`** для указателей, чтобы компилятор мог оптимизировать (при отсутствии алиасинга).
- **Разворачивание циклов** (ручное или `#pragma unroll`, зависящее от компилятора), вынос инвариантов из цикла.
- Избыточные **регистры** → spill в медленную память: упрощать ядро, разбивать на несколько.

### 34.9. Константная память и данные только для чтения

- Небольшие таблицы коэффициентов, читаемые всеми work-item одновременно, — в **`__constant`** (кэшируется, broadcast быстр). Если work-item читают **разные** адреса — выигрыш пропадает.
- Для 2D-данных с локальностью по обеим осям — **image-объекты** (текстуры) с аппаратной фильтрацией и кэшем (в слайдах упомянуты как «image manipulation»).

### 34.10. Перекрытие вычислений и передач

- Несколько очередей (или out-of-order) + неблокирующие передачи и события → **копирование следующей порции данных идёт параллельно с вычислением предыдущей** (double buffering).
- Закреплённая память (`ALLOC_HOST_PTR`) ускоряет передачу и необходима для реального перекрытия.
- `clEnqueueMapBuffer` вместо `Read/Write` — на интегрированных GPU и CPU позволяет обойтись **без копирования**.

### 34.11. Накладные расходы запуска

- Запуск ядра имеет фиксированные расходы (микросекунды–десятки микросекунд). Очень короткие ядра выгоднее **объединять (kernel fusion)**.
- Компиляция программы — один раз; ядра/буферы **создавать один раз** и переиспользовать, не пересоздавая в цикле.
- Не вызывать `clFinish` после каждой команды без необходимости — теряется конвейеризация.

### 34.12. Сравнение с CUDA — нюансы портирования

- **Компиляция:** CUDA — `nvcc` заранее (офлайн); OpenCL — **во время выполнения** (JIT).
- **Переносимость:** OpenCL работает на CPU/GPU/ускорителях разных производителей (платформа/устройство выбираются в рантайме); CUDA — только NVIDIA.
- **Запуск:** CUDA `kernel<<<grid, block>>>` — число **блоков** и размер блока; OpenCL — **общее** число work-item (`global`) и размер группы (`local`).
- **Указатели:** `cudaMalloc` даёт обычный указатель; `cl_mem` — дескриптор.
- **Хостовый API:** в OpenCL много «церемонии» (платформа → устройство → контекст → очередь → программа → ядро → аргументы), в CUDA runtime API проще. Но по структуре хост-кода OpenCL близок к **CUDA Driver API**.
- **Warp ≠ wavefront по размеру** (32 против 64 у старых AMD): код, завязанный на размер warp, **непереносим**.
- **Адресные пространства** в OpenCL — явные в типе указателя (`__global float*`), в CUDA — выводятся автоматически.
- **Термины памяти** (см. §16): **Shared↔Local**, **Local↔Private** — частая путаница.

---

## 35. «Каверзные» вопросы с ответами (тренировка)

1. **Что произойдёт, если `global_work_size = 10`, а `local_work_size = 4` (OpenCL 1.x)?** — Ошибка `CL_INVALID_WORK_GROUP_SIZE`: 10 не делится на 4 нацело. Решение: округлить global до 12 и добавить в ядро `if (i < n)`.
2. **Блокирует ли `clEnqueueNDRangeKernel` хост?** — Нет, он только ставит команду в очередь. Ждать надо `clFinish`/`clWaitForEvents` или блокирующим чтением.
3. **Нужен ли `clFinish`, если следом идёт `clEnqueueReadBuffer` с `CL_TRUE`?** — Не обязательно: в in-order очереди чтение выполнится после ядра и вернёт управление, когда данные скопированы.
4. **Что будет, если прочитать буфер неблокирующим чтением и сразу использовать `res_cpu`?** — Данных может ещё не быть (гонка); нужно дождаться события.
5. **Почему ядро `sum` корректно работает при `gridSize = 10` и `n = 1000`?** — Цикл с шагом `get_global_size(0)`: каждый work-item обрабатывает `id, id+10, id+20, …`.
6. **Что надо сделать с `res` перед запуском `sum`?** — Инициализировать нулём (например, создать буфер с `COPY_HOST_PTR` из обнулённого массива), иначе `atomic_add` добавит к мусору.
7. **Почему `atomic_add` нужен?** — Иначе гонка данных при одновременной записи в `res[0]`.
8. **Что вернёт `get_global_id(1)` при одномерном запуске?** — 0. А `get_global_size(1)` — 1.
9. **Чему равен `get_global_id(0)` через другие функции?** — `get_group_id(0) * get_local_size(0) + get_local_id(0)` (+ `get_global_offset(0)`).
10. **Чем отличается `get_global_size` от `get_num_groups`?** — Первая — число work-item, вторая — число групп. `global_size = num_groups * local_size`.
11. **Какой размер у `float3`?** — 16 байт (как `float4`).
12. **Что вернёт `abs` для `int`?** — `uint`.
13. **Что вернёт `isequal` для скаляров и для векторов?** — Скаляры: 1/0. Векторы: −1/0.
14. **Что вернёт `any((int4)(0, 0, -1, 0))`?** — 1 (старший бит у третьей компоненты равен 1).
15. **Что вернёт `all((int4)(-1, -1, 0, -1))`?** — 0 (старший бит нулевой у третьей компоненты).
16. **`float4 a = (float4)(1,2,3,4); a.xx = (float2)(5,6);` — допустимо?** — Нет, повторы компонент слева запрещены.
17. **Допустимо ли `float2 p; p.z = 1.0f;`?** — Нет, ошибка компиляции (у `float2` нет `z`).
18. **Можно ли `float4 v = some_int4;`?** — Нет, неявное преобразование между векторными типами запрещено; использовать `convert_float4()` (значения) или `as_float4()` (биты).
19. **Чем `convert_` отличается от `as_`?** — `convert_` преобразует значения (1 → 1.0f), `as_` переинтерпретирует биты без изменения.
20. **Что даёт `(float4)(2.0f)`?** — Вектор `(2,2,2,2)`.
21. **В чём разница `step` и `smoothstep`?** — `step` — скачок (0 при `x<edge`, иначе 1), `smoothstep` — плавный переход по полиному Эрмита между `edge0` и `edge1`.
22. **Чему равен `step(2.0f, 2.0f)`?** — 1.0 (при `x == edge` возвращается 1).
23. **Чем `remainder` отличается от `fmod`?** — `remainder` использует ближайшее целое частного (результат может быть отрицательным), `fmod` — усечение (знак как у делимого).
24. **Чем `round` отличается от `rint`?** — `round`: половины от нуля (2.5 → 3); `rint`: к чётному (2.5 → 2).
25. **Чем `mad` отличается от `fma`?** — `mad` — приближённо, реализация вправе пожертвовать точностью ради скорости; `fma` — точное fused-вычисление с одним округлением.
26. **Какой порядок аргументов у `atan2`?** — `atan2(y, x)`.
27. **Для каких размеров вектора определены `dot`/`length`?** — Для `float`, `float2`, `float3`, `float4` (не для 8 и 16).
28. **Почему `1.0` в ядре — плохая идея?** — Это `double`; без `cl_khr_fp64` ошибка, с ним — медленнее. Нужно `1.0f`.
29. **Можно ли поставить `barrier` внутри `if (lid < 5)`?** — Нет: барьер должен достигаться всеми work-item группы → зависание/UB.
30. **Можно ли синхронизировать разные work-group внутри ядра?** — Нет; только завершением ядра и запуском следующего.
31. **Гарантируется ли порядок выполнения work-group?** — Нет, и не все группы выполняются одновременно.
32. **Что произойдёт при вызове `clGetEventProfilingInfo`, если очередь создана с `properties = 0`?** — Ошибка `CL_PROFILING_INFO_NOT_AVAILABLE`.
33. **Что произойдёт, если вызвать `clGetEventProfilingInfo` до завершения команды?** — Тоже ошибка недоступности; сначала `clWaitForEvents`.
34. **В каких единицах время профилирования?** — Наносекунды (`cl_ulong`).
35. **Почему первый запуск ядра медленнее?** — Ленивое выделение/копирование буферов, JIT и инициализация драйвера; поэтому делают прогрев.
36. **Что будет, если в `clSetKernelArg` передать `int` вместо `float`?** — Ошибки не будет, но байты интерпретируются как `float` → неверный результат.
37. **Как передать размер `__local`-массива в ядро?** — `clSetKernelArg(kernel, idx, size_in_bytes, NULL)`.
38. **Зачем `errcode_ret`?** — Чтобы получить код ошибки от функций, возвращающих объект; при `NULL` ошибка теряется.
39. **Можно ли использовать указатель `cl_mem` на хосте как массив?** — Нет, это дескриптор.
40. **Что значит `CL_MEM_USE_HOST_PTR` и когда он выгоден?** — Буфер использует память хоста; выгоден на CPU/интегрированных GPU (без копирования). На дискретных GPU драйвер обычно копирует сам.
41. **Можно ли в OpenCL 1.x объявить глобальную переменную `__global int g;` вне функций?** — Нет; на уровне программы допустимы только `__constant`. (2.0+ — можно.)
42. **Почему `__local`-массив содержит мусор?** — Локальная память не инициализируется; нужно заполнять и вызывать `barrier`.
43. **Что быстрее: `x / 2.0f` или `x * 0.5f`?** — Умножение (деление дороже).
44. **Как избежать bank conflict при доступе к `tile[16][16]`?** — Дополнить строку: `tile[16][17]`.
45. **Почему в CUDA `gridDim.x` ≠ `get_global_size(0)`?** — `gridDim.x` — число блоков, `get_global_size(0)` — число work-item (= `gridDim.x * blockDim.x`).
46. **Можно ли в OpenCL C использовать рекурсию?** — Нет.
47. **Что делает опция `-cl-fast-relaxed-math`?** — Включает быстрые, менее точные варианты математики (возможна потеря корректности NaN/Inf).
48. **Что нужно для `double` в ядре?** — Расширение `cl_khr_fp64` (`#pragma OPENCL EXTENSION cl_khr_fp64 : enable`) и поддержка устройства.
49. **Почему размер `long` в ядре и на хосте может не совпасть, если на хосте писать `long`?** — В OpenCL C `long` всегда 64 бита, а в C на Windows — 32; на хосте использовать `cl_long`.
50. **Что быстрее на GPU — один work-item на элемент или цикл по куску?** — Как правило, много work-item и коалесцированный доступ; на CPU чаще выгоднее крупные непрерывные куски.

---

## 36. Мини-шпаргалка по «подводным камням» (повторить в последнюю очередь)

- Ядро **асинхронно**, ждать надо явно (`clFinish` / событие / блокирующее чтение).
- Профилирование = флаг в очереди + событие + ожидание; время в **нс**.
- `global % local == 0` (в 1.x); `local ≤ MAX_WORK_GROUP_SIZE`; кратно 32/64.
- `barrier` — **все** work-item группы, вне условий; между группами синхронизации нет.
- Атомарные операции — **32-бит int**, на один адрес сериализуются; `float` атомарно в 1.x нельзя.
- `1.0` — double, `1.0f` — float; `long` = 64 бит; `float3` = 16 байт.
- Вектор: сравнения → **−1/0**; операции **покомпонентные**; неявных преобразований между векторными типами **нет**; swizzle слева **без повторов**.
- `step(edge, x)`, `atan2(y, x)`, `clamp(x, min, max)` — **порядок аргументов!**
- Geometric-функции — только до `float4`; `cross` — `float3/4`.
- `abs(int) → uint`; `remainder ≠ fmod`; `round ≠ rint`; `mad ≠ fma`.
- Local (CUDA) = Private (OpenCL); Shared (CUDA) = Local (OpenCL).
- Передачи по PCIe — главный враг; коалесцирование и локальная память — главные друзья.

Удачи на тесте! 🍀
