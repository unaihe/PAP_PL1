#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>
//fase2
#define MAX_MATRICULA 16
//fase3 (definimos las opciones para no tener que comparar strings)
// Columnas
#define COL_DEP 0
#define COL_ARR 1
#define COL_WEATHER 2
// Operaciones
#define OP_MAX 0
#define OP_MIN 1
// Variantes del ejercicio
#define VAR_SIMPLE 1
#define VAR_BASICA 2
#define VAR_INTERMEDIA 3
#define VAR_PATRON 4
// fase4 (Mas o menos el numero de ids que hay en el aeropuerto)
#define TIPO_ORIGEN 0
#define TIPO_DESTINO 1

__constant__ int d_umbral;
__constant__ int d_umbral_f02;
__device__ int d_contador_f02;

void cargarDataset(std::string ruta,
    std::vector<float>& dep_delay,
    std::vector<float>& arr_delay,
    std::vector<std::string>& tail_num, 
    std::vector<float>& weather_delay,
    std::vector<int>& origin_id,
    std::vector<int>& dest_id,
    std::unordered_map<int, std::string>& mapa_aeropuertos);

void resetYCargar(std::string ruta,
    std::vector<float>& dep_delay,
    std::vector<float>& arr_delay,
    std::vector<std::string>& tail_num,
    std::vector<float>& weather_delay,
    std::vector<int>& origin_id,
    std::vector<int>& dest_id,
    std::unordered_map<int, std::string>& mapa_aeropuertos,
    bool& estado);

void cargarFase01(std::vector<float>& retrasos, int umbral);

void cargarFase02(std::vector<std::string>& h_tail_num, std::vector<float>& retrasos, int umbral);

void cargarFase03(std::vector<float>& retrasos, int operacion_elegida, int variante_elegida);

void cargarFase04(std::vector<int>& vectorElegido, int umbral_hist, std::unordered_map<int, std::string>& mapa);

void configurarGridEstandar(int n, int& blocks, int& threads);

__global__ void fase01(float *d_retraso, int num_vuelos) {
    bool signo = true;
    // Calculamos el ID global
    int posicion = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Comprobacion de si el hilo se pasa del total de vuelos
    if (posicion >= num_vuelos) {
        return;
    }

    // Obtenemos el valor del retraso para este hilo
    float valor = d_retraso[posicion];

    // Siu es NaN se ignora el valor
    if (isnan(valor)) return;

    // Comprobamos si es positivo o negativo
    if (d_umbral < 0) {
        signo = false;
    }
    // Comprobacion del umbral pra positivos y negativos
    if (valor >= d_umbral && signo) {
        printf("Hilo #%d: Retraso detectado de %.0f minutos\n", posicion, valor);
    }
    else if (valor <= d_umbral && !signo) {
        printf("Hilo #%d: Adelanto detectado de %.0f minutos\n", posicion, valor);
    }
}

__global__ void fase02(float *d_arr_ent, float *d_arr_sal, char *d_tail_ent, char *d_tail_sal, int num_vuelos) {
    //creamos la variable signo que usaremos para comprobar si se trata de un adelanto o un retraso
    bool signo;
    //Y cumple para ver si es solucion o no
    bool cumple = false;
    //Calculamos la posicion del hilo y comprobamos que este dentro del vector
    int pos = (blockDim.x*blockIdx.x)+threadIdx.x;
    if (pos >= num_vuelos) {
        return;
    }
    //Cogemos el retraso del hilo
    float retraso = d_arr_ent[pos];
    //Comprobamos si es retraso o adelantos
    if (retraso >= 0) {
        signo = true;
    }
    else {
        signo = false;
    }
    if (d_umbral_f02 >= 0) {
        if (retraso >= d_umbral_f02) cumple = true;
    }
    else {
        if (retraso <= d_umbral_f02) cumple = true;
    }
    //Lo implementamos de la siguiente forma para no tener repetida dos veces la suma en el if y en if else
    if (cumple) {
        int mi_indice = atomicAdd(&d_contador_f02, 1);
        d_arr_sal[mi_indice] = retraso;
        for (int i = 0; i < MAX_MATRICULA; i++) {
            d_tail_sal[mi_indice * MAX_MATRICULA + i] = d_tail_ent[pos * MAX_MATRICULA + i];
        }
    }

}

__global__ void fase03_simple(float *d_datos, int *d_resultado, int num_vuelos, int operacion) {
    //Calculamos la posicion del hilo y comprobamos que este dentro del vector
    int pos = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (pos < num_vuelos) {
        float dato = d_datos[pos];
        if (isnan(dato)) return;

        // Truncado a int 
        int valor = (int)dato;

        if (operacion == OP_MAX) {
            atomicMax(d_resultado, valor);
        }
        else {
            atomicMin(d_resultado, valor);
        }
    }
}

__global__ void fase03_basica(float* d_datos, int* d_resultado, int num_vuelos, int operacion) {
    //Memoria compartida
    extern __shared__ int sh_datos[];
    //La usaremos para la memoria compartida
    int posicion_hilo = threadIdx.x;
    //Posicion en el vector completo
    int pos = (blockDim.x * blockIdx.x) + threadIdx.x;
    //Comprobamos que la posición es valida
    if (pos >= num_vuelos) {
        sh_datos[posicion_hilo] = (operacion == OP_MAX) ? -999999 : 999999;
    }
    else {
        float dato = d_datos[pos];
        // Si el dato es NaN, ponemos un valor neutro para no estropear el max/min
        if (isnan(dato)) {
            sh_datos[posicion_hilo] = (operacion == OP_MAX) ? -999999 : 999999;
        }
        else {
            sh_datos[posicion_hilo] = (int)dato; // Truncamos el dato
        }
    }
    __syncthreads();
    //Continuar con logica del vecino de izquierda y derecha
    if (pos < num_vuelos) {
        int mi_valor = sh_datos[posicion_hilo];
        //Variable que usaremos para saber si es posible que sea max o min o deshechamos
        bool sigo_siendo_candidato = true;

        //Mirar a la izquierda
        if (posicion_hilo > 0) {
            int vecino_izq = sh_datos[posicion_hilo - 1];
            if (operacion == OP_MAX) {
                if (vecino_izq > mi_valor) sigo_siendo_candidato = false;
            }
            else {
                if (vecino_izq < mi_valor) sigo_siendo_candidato = false;
            }
        }

        //Mirar a la derecha (solo si aún no me han descartado)
        if (sigo_siendo_candidato && posicion_hilo < blockDim.x - 1) {
            int vecino_der = sh_datos[posicion_hilo + 1];
            if (operacion == OP_MAX) {
                if (vecino_der > mi_valor) sigo_siendo_candidato = false;
            }
            else {
                if (vecino_der < mi_valor) sigo_siendo_candidato = false;
            }
        }

        //El que no ha sido descartado por sus vecinos hace el atomic
        if (sigo_siendo_candidato) {
            if (operacion == OP_MAX) {
                atomicMax(d_resultado, mi_valor);
            }
            else {
                atomicMin(d_resultado, mi_valor);
            }
        }
    }
}

__global__ void fase03_intermedia(float* d_datos, int* d_resultado, int num_vuelos, int operacion) {
    //Memoria compartida
    extern __shared__ int sh_datos[];
    //La usaremos para la memoria compartida
    int posicion_hilo = threadIdx.x;
    //Posicion en el vector completo
    int pos = (blockDim.x * blockIdx.x) + threadIdx.x;

    //Cargamos los datos a la memoria compartida, los que estan dentro y fuera porque tambien se haran comprobaciones en el ultimo valor
    if (pos < num_vuelos) {
        float val = d_datos[pos];
        sh_datos[posicion_hilo] = isnan(val) ? ((operacion == OP_MAX) ? -999999 : 999999) : (int)val;
    }
    else {
        sh_datos[posicion_hilo] = (operacion == OP_MAX) ? -999999 : 999999;
    }

    //Esperamos a que todos los hilos carguen sus datos
    __syncthreads();

    //Solo cogemos uno de cada dos hilos para hacer los calculos de quien es mayor o menor
    // Añadimos: && posicion_hilo + 1 < blockDim.x para no leer fuera del array
    if (posicion_hilo % 2 == 0 && (posicion_hilo + 1) < blockDim.x) {
        int mi_val = sh_datos[posicion_hilo];
        int vecino = sh_datos[posicion_hilo + 1];

        if (operacion == OP_MAX) {
            int ganador = (mi_val > vecino) ? mi_val : vecino;
            atomicMax(d_resultado, ganador);
        }
        else { // Si no es max es min
            int ganador = (mi_val < vecino) ? mi_val : vecino;
            atomicMin(d_resultado, ganador);
        }
    }
}

__global__ void fase03_patron(float* d_datos, int* d_resultado, int num_vuelos, int operacion) {
    //Memoria compartida
    extern __shared__ int sh_datos[];
    //La usaremos para la memoria compartida
    int posicion_hilo = threadIdx.x;
    //Posicion en el vector completo
    int pos = (blockDim.x * blockIdx.x) + threadIdx.x;
    //Cargamos los datos a la memoria compartida, los que estan dentro y fuera porque tambien se haran comprobaciones en el ultimo valor
    if (pos < num_vuelos) {
        float val = d_datos[pos];
        sh_datos[posicion_hilo] = isnan(val) ? ((operacion == OP_MAX) ? -999999 : 999999) : (int)val;
    }
    else {
        sh_datos[posicion_hilo] = (operacion == OP_MAX) ? -999999 : 999999;
    }
    //Esperamos a que todos los hilos carguen sus datos
    __syncthreads();
    
    //Creamos el primer salto es decir la mitad del bloque y se irá dividiendo entre dos en cada paso
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (posicion_hilo < s) {
            int mi_val = sh_datos[posicion_hilo];
            int mi_val_salto = sh_datos[posicion_hilo+s];
            if (mi_val < mi_val_salto && operacion == OP_MAX) {
                sh_datos[posicion_hilo] = mi_val_salto;
            }
            else if (mi_val > mi_val_salto && operacion == OP_MIN) {
                sh_datos[posicion_hilo] = mi_val_salto;
            }
        }
        //Esperamos a que todos los hilos hagan el max o min entre ellos y seguimos a la siguiente iteracion
        __syncthreads();
    }
    if (posicion_hilo == 0) {
        if (operacion == OP_MAX) {
            atomicMax(d_resultado, sh_datos[0]);
        }
        else {
            atomicMin(d_resultado, sh_datos[0]);
        }
    }
}

__global__ void fase04(int* d_ids, int* d_histograma, int num_vuelos, int tam_max) {
    // Calculamos la posición global del hilo
    int pos = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Si el hilo está dentro del rango de vuelos
    if (pos < num_vuelos) {
        int id = d_ids[pos];
        // Solo sumamos si el ID es válido y cabe en nuestro array
        if (id >= 0 && id < tam_max) {
            // atomicAdd evita que dos hilos sobrescriban el mismo valor a la vez
            atomicAdd(&d_histograma[id], 1);
        }
    }
}

int main()
{
    // Traza inicial 
    std::cout << "Iniciando aplicacion PAP - PL1..." << std::endl;

    // Vectores para almacenar las columnas que usaremos
    std::vector<float> h_dep_delay;   
    std::vector<float> h_arr_delay;   
    std::vector<std::string> h_tail_num;
    std::vector<float> h_weather_delay;  
    std::vector<int> h_origin_id;        
    std::vector<int> h_dest_id;
    std::unordered_map<int, std::string> mapa_aeropuertos;

    //Ruta al csv con los datos
    std::string ruta_csv;
    bool datos_listos = false;
    //Opcion para insertar ruta por teclado o elegir la generica
    std::cout << "Introduzca la ruta base del dataset (Enter para ruta por defecto): ";
    std::getline(std::cin, ruta_csv);

    if (ruta_csv.empty()) {
        ruta_csv = "C:\\Users\\Unai\\OneDrive\\Documentos\\Airline_dataset.csv";
    }

    // cargado de datos
    resetYCargar(ruta_csv, h_dep_delay, h_arr_delay, h_tail_num, h_weather_delay, h_origin_id, h_dest_id, mapa_aeropuertos, datos_listos);

    
    std::cout << "DEBUG: Vuelos totales en el vector: " << h_origin_id.size() << std::endl;
    char opcion;
    do {
        std::cout << "\nMenu de opciones:" << std::endl;
        std::cout << "(0) Cambio de ruta" << std::endl;
        std::cout << "(1) Retraso en salida" << std::endl;
        std::cout << "(2) Retraso en llegada" << std::endl;
        std::cout << "(3) Reduccion de retraso" << std::endl;
        std::cout << "(4) Histograma de aeropuertos" << std::endl;
        std::cout << "(x) Salir" << std::endl;
        std::cout << "Seleccione una opcion: ";
        std::cin >> opcion;

        switch (opcion) {
        case '0': {
            std::cout << "Introduce la nueva ruta: ";
            std::cin.ignore(1000, '\n');
            std::getline(std::cin, ruta_csv);

            //Carga automatica tras el cambio
            resetYCargar(ruta_csv, h_dep_delay, h_arr_delay, h_tail_num, h_weather_delay, h_origin_id, h_dest_id, mapa_aeropuertos, datos_listos);

            break;
        }
        case '1': {
            if (!datos_listos)
            { 
                std::cout << "[!] Error: No hay datos cargados. Comprueba la ruta en la opcion (0)." << std::endl;
            }
            else {
                // Retraso en despegues 
                int umbral;
                std::string entrada;

                // Limpiamos el buffer del '1' que elegimos en el menú para que no interfiera
                std::cin.ignore(1000, '\n');

                // Bucle para asegurar que el usuario introduce un valor correcto y no deje el campo vacio
                while (true) {
                    std::cout << "Introduce el umbral de retraso (minutos): ";

                    // Leemos la linea completa para evitar errores si el usuario mete letras o espacios
                    if (!std::getline(std::cin, entrada)) continue;

                    // Si solo pulsa Enter sin escribir nada, volvemos a preguntar
                    if (entrada.empty()) continue;

                    try {
                        // Intentamos convertir el texto a numero entero
                        umbral = std::stoi(entrada);
                        break; // Si la conversion tiene exito, salimos del bucle de validacion
                    }
                    catch (...) {
                        // Si salta un error (letras, simbolos...), avisamos al usuario
                        std::cout << "Error: Debes introducir un numero entero valido." << std::endl;
                    }
                }

                std::cout << "Ejecutando Fase 01 con umbral: " << umbral << "..." << std::endl;
                //Llamada a la función con el umbral determinado
                //Llamamos a cargarFase porque es la que llama a fase01 , prepara los espacios de memoria y copia los vectores
                cargarFase01(h_dep_delay, umbral);

                break;
            }
        }
        case '2': {
            if (!datos_listos)
            {
                std::cout << "[!] Error: No hay datos cargados. Comprueba la ruta en la opcion (0)." << std::endl;
            }
            else {
                // Retraso en aterrizajes 
                //Variables para el umbral
                int umbral_llegada;
                std::string entrada_f2;

                //Limpiamos el buffer por si acaso
                std::cin.ignore(1000, '\n');

                //Bucle de validación (Igual que en Fase 1)
                while (true) {
                    std::cout << "\n--- FASE 02: Analisis de Llegadas ---" << std::endl;
                    std::cout << "Introduce el umbral (positivo para retraso, negativo para adelanto): ";

                    if (!std::getline(std::cin, entrada_f2)) continue;
                    if (entrada_f2.empty()) continue;

                    try {
                        umbral_llegada = std::stoi(entrada_f2);
                        break; // Si es un número válido, salimos del bucle
                    }
                    catch (...) {
                        std::cout << "Error: Por favor, introduce un numero entero (ej: 1440 o -30)." << std::endl;
                    }
                }
                std::cout << "Ejecutando Fase 02 con umbral " << umbral_llegada << "..." << std::endl;

                //Llamada a la función cargarFase02 (igual que con la fase1)
                cargarFase02(h_tail_num, h_arr_delay, umbral_llegada);
                break;
            }
        }
        case '3': {
            if (!datos_listos)
            {
                std::cout << "[!] Error: No hay datos cargados. Comprueba la ruta en la opcion (0)." << std::endl;
            }
            else {
                //Máximos/Mínimos 
                int col_sel, op_sel, var_sel;

                std::cout << "\n--- FASE 03: Reduccion de Retraso ---" << std::endl;

                //Selección de Columna
                std::cout << "Seleccione columna (" << COL_DEP << ":DEP, " << COL_ARR << ":ARR, " << COL_WEATHER << ":WEA): ";
                if (!(std::cin >> col_sel) || col_sel < 0 || col_sel > 2) {
                    std::cout << "Error: Seleccion de columna no valida." << std::endl;
                    std::cin.clear(); std::cin.ignore(1000, '\n');
                    break;
                }

                //Selección de Operación
                std::cout << "Seleccione operacion (" << OP_MAX << ":Max, " << OP_MIN << ":Min): ";
                if (!(std::cin >> op_sel) || (op_sel != OP_MAX && op_sel != OP_MIN)) {
                    std::cout << "Error: Operacion no valida." << std::endl;
                    std::cin.clear(); std::cin.ignore(1000, '\n');
                    break;
                }

                //Selección de Variante
                std::cout << "Seleccione variante (1:Simple, 2:Basica, 3:Intermedia, 4:Patron): ";
                if (!(std::cin >> var_sel) || var_sel < 1 || var_sel > 4) {
                    std::cout << "Error: Variante no valida." << std::endl;
                    std::cin.clear(); std::cin.ignore(1000, '\n');
                    break;
                }

                //Pasamos el vector correcto según la elección
                if (col_sel == COL_DEP) {
                    cargarFase03(h_dep_delay, op_sel, var_sel);
                }
                else if (col_sel == COL_ARR) {
                    cargarFase03(h_arr_delay, op_sel, var_sel);
                }
                else if (col_sel == COL_WEATHER) {
                    cargarFase03(h_weather_delay, op_sel, var_sel);
                }

                break;
            }
        }
        case '4': {
            if (!datos_listos)
            {
                std::cout << "[!] Error: No hay datos cargados. Comprueba la ruta en la opcion (0)." << std::endl;
            }
            else {
                // Histograma 
                int tipo_sel; // 0 para Origen, 1 para Destino
                int umbral_hist;

                std::cout << "\n--- FASE 04: Histograma de Aeropuertos ---" << std::endl;

                //Pedimos el tipo
                std::cout << "Seleccione tipo de aeropuerto (0: ORIGEN, 1: DESTINO): ";
                while (!(std::cin >> tipo_sel) || (tipo_sel != 0 && tipo_sel != 1)) {
                    std::cout << "Error: Seleccione 0 o 1: ";
                    std::cin.clear();
                    std::cin.ignore(1000, '\n');
                }

                //Pedimos el umbral
                std::cout << "Introduce el umbral minimo de ocurrencias para mostrar: ";
                while (!(std::cin >> umbral_hist) || umbral_hist < 0) {
                    std::cout << "Error: Introduzca un numero positivo: ";
                    std::cin.clear();
                    std::cin.ignore(1000, '\n');
                }

                //Llamamos a la función cargadora pasando el vector correspondiente
                if (tipo_sel == 0) {
                    std::cout << "Generando histograma de salidas (ORIGIN)..." << std::endl;
                    cargarFase04(h_origin_id, umbral_hist, mapa_aeropuertos);
                }
                else {
                    std::cout << "Generando histograma de llegadas (DEST)..." << std::endl;
                    cargarFase04(h_dest_id, umbral_hist, mapa_aeropuertos);
                }
                break;
            }
        }
        case 'x':
            std::cout << "Saliendo..." << std::endl;
            break;
        default:
            std::cout << "Opcion no valida." << std::endl;
        }
    } while (opcion != 'x');

    return 0;
}

// Función para cargar y limpiar los datos del CSV
void cargarDataset(std::string ruta,
    std::vector<float>& dep_delay,
    std::vector<float>& arr_delay,
    std::vector<std::string>& tail_num,
    std::vector<float>& weather_delay,
    std::vector<int>& origin_id,
    std::vector<int>& dest_id,
    std::unordered_map<int, 
    std::string>& mapa_aeropuertos) {

    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "Error crítico: No se puede abrir el archivo en " << ruta << std::endl;
        return;
    }

    std::string linea;
    // Saltamos la cabecera del CSV 
    std::getline(archivo, linea);

    // Traza de ejecución requerida 
    std::cout << "Procesando lineas del dataset" << std::endl;

    int id_origen_temp, id_destino_temp;

    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        std::string celda;
        int columna = 0;

        while (std::getline(ss, celda, ',')) {
            // Columna 3: Matrícula (TAIL_NUM) para Fase 02 
            if (columna == 3) {
                tail_num.push_back(celda);
            }
            // Columna 5: ID Aeropuerto Origen (ORIGIN_SEQ_ID) 
            else if (columna == 5) {
                if (!celda.empty()) {
                    int id = (int)std::stof(celda);
                    origin_id.push_back(id);
                    id_origen_temp = id; // Guardamos el ID para asociarlo luego al nombre
                }
            }
            // Columna 6: Nombre Aeropuerto Origen (JFK)
            else if (columna == 6) {
                mapa_aeropuertos[id_origen_temp] = celda;
            }
            // Columna 7: ID Aeropuerto Destino (DEST_SEQ_ID)
            else if (columna == 7) {
                if (!celda.empty()) {
                    int id = (int)std::stof(celda);
                    dest_id.push_back(id);
                    id_destino_temp = id;
                }
            }
            // Columna 8: Nombre Aeropuerto Destino (PHX)
            else if (columna == 8) {
                mapa_aeropuertos[id_destino_temp] = celda;
            }
            // Columna 10: Retraso Salida (DEP_DELAY) para Fase 01 
            else if (columna == 10) {
                if (celda.empty()) dep_delay.push_back(NAN); // Limpieza NAN 
                else dep_delay.push_back(std::stof(celda));
            }
            // Columna 12: Retraso Llegada (ARR_DELAY) para Fase 02 
            else if (columna == 12) {
                if (celda.empty()) arr_delay.push_back(NAN); // Limpieza NAN 
                else arr_delay.push_back(std::stof(celda));
            }
            // Columna 13: Retraso Meteorológico (WEATHER_DELAY) para Fase 03 
            else if (columna == 13) {
                if (celda.empty()) weather_delay.push_back(NAN); // Limpieza NAN 
                else weather_delay.push_back(std::stof(celda));
            }
            columna++;
        }
    }
    archivo.close();
    // Traza final requerida 
    std::cout << "Carga finalizada con exito." << std::endl;
}

void cargarFase01(std::vector<float>& retrasosSalida, int umbral) {

    //Calculamos el numero de vuelos del vector para poder guardar el espacio necesario en memoria
    int num_vuelos = retrasosSalida.size();
    //Creamos una variable puntero donde se iniciará el vector en la GPU
    float* d_retrasos;
    //Calculamos el numero de hilos y bloques
    int bloques, hilosPorBloque;
    configurarGridEstandar(num_vuelos, bloques, hilosPorBloque);

    std::cout << "[GPU] Procesando " << num_vuelos << " vuelos..." << std::endl;

    //Guardamos la memoria necesaria en la gpu a través del tamaño del vector calculado con el .size
    cudaMalloc(&d_retrasos, num_vuelos * sizeof(float));
    //Copiamos a la memoria de la GPU el vector entero de retrasos
    cudaMemcpy(d_retrasos, retrasosSalida.data(), num_vuelos * sizeof(float), cudaMemcpyHostToDevice);

    //Ponemos la variable en memoria
    cudaMemcpyToSymbol(d_umbral, &umbral, sizeof(int));

    //Llamamos a la función de la GPU para que procese todos los datos
    fase01 <<< bloques, hilosPorBloque >>> (d_retrasos, num_vuelos);

    //Sincronizamos todos los hilos(para comprobar que todos han terminado)
    cudaDeviceSynchronize();
    //Liberamos la memoria de la GPU
    cudaFree(d_retrasos);

    std::cout << "[GPU] Analisis finalizado." << std::endl;
}

void cargarFase02(std::vector<std::string>& h_tail_num, std::vector<float>& retrasosLlegada, int umbral) {
    int num_vuelos = h_tail_num.size();

    //Buscamos la matrícula más larga (Se utilizará como dato para guardar memoria necesaria)
    int max_long = 0;
    for (const std::string& s : h_tail_num) {
        if (s.length() > max_long) max_long = s.length();
    }
    //Cambiamos el vector de string a char, debido a que no existe string en la gpu
    //Con el ,0 nos aseguramos de que si una mátricula no ocupa el tamaño maximo se rellena con 0s
    std::vector<char> h_tail_num_c(num_vuelos * MAX_MATRICULA, 0);
    //Utilizamos un bucle para iterar sobre cada una de las mátriculas y otro para iterar sobre cada letra de la matricula
    for (int i = 0; i < num_vuelos; ++i) {
        for (int j = 0; j < h_tail_num[i].length() && j < MAX_MATRICULA; ++j) {
            h_tail_num_c[i * MAX_MATRICULA + j] = h_tail_num[i][j];
        }
    }
    //Creamos los punteros que usaremos a la hora de reservar la memoria de entrada en la gpu
    float* d_arr_delay_ent;
    char* d_tail_num_ent;
    //Y creamos los mismos para los vectores que devolveremos con los resultados
    float* d_arr_delay_sal;
    char* d_tail_num_sal;

    //Guardamos la memoria necesaria en gpu para los vectores de matriculas y de llegadas
    cudaMalloc(&d_arr_delay_ent,num_vuelos*sizeof(float));
    cudaMalloc(&d_tail_num_ent, num_vuelos * MAX_MATRICULA * sizeof(char));
    //También reservamos para los vectores con las soluciones (mismo tamaño por si el peor caso todos son solucion)
    cudaMalloc(&d_arr_delay_sal, num_vuelos * sizeof(float));
    cudaMalloc(&d_tail_num_sal, num_vuelos * MAX_MATRICULA * sizeof(char));
    //Copiamos los datos a la gpu
    cudaMemcpy(d_arr_delay_ent,retrasosLlegada.data(), num_vuelos * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tail_num_ent, h_tail_num_c.data(), num_vuelos * MAX_MATRICULA * sizeof(char), cudaMemcpyHostToDevice);

    //Calculamos el numero de hilos y bloques
    int bloques, hilosPorBloque;
    configurarGridEstandar(num_vuelos, bloques, hilosPorBloque);

    //Ponemos el contador de la GPU a cero (lo usaremos para que no haya condiciones de carrera)
    int cero = 0;
    cudaMemcpyToSymbol(d_contador_f02, &cero, sizeof(int));
    //Pasamos el umbral que el usuario eligió 
    cudaMemcpyToSymbol(d_umbral_f02, &umbral, sizeof(int));

    //Llamamos a la función de la GPU para que procese todos los datos
    fase02 <<< bloques, hilosPorBloque >>> (d_arr_delay_ent,d_arr_delay_sal,d_tail_num_ent,d_tail_num_sal,num_vuelos);
    
    //Sincronizamos todos los hilos(para comprobar que todos han terminado)
    cudaDeviceSynchronize();

    //Miramos el numero de soluciones que hay(longitud del array solucion) 
    int vuelos_solucion;
    cudaMemcpyFromSymbol(&vuelos_solucion, d_contador_f02, sizeof(int));

    if (vuelos_solucion > 0) {
        //Preparamos vectores en la CPU para recibir los datos
        std::vector<float> h_tiempos_res(vuelos_solucion);
        std::vector<char> h_mats_res(vuelos_solucion * MAX_MATRICULA);

        //Copiamos los resultados de la salida de la GPU a nuestros vectores de la CPU
        cudaMemcpy(h_tiempos_res.data(), d_arr_delay_sal, vuelos_solucion * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mats_res.data(), d_tail_num_sal, vuelos_solucion * MAX_MATRICULA * sizeof(char), cudaMemcpyDeviceToHost);

        //Mostramos los resultados por pantalla
        std::cout << "Se han detectado " << vuelos_solucion << " vuelos:" << std::endl;
        for (int i = 0; i < vuelos_solucion; i++) {
            // Usamos el puntero a la posición i-ésima para que printf lo lea como string
            printf("Vuelo [%d]: Matricula %s | Tiempo: %.0f min\n", i + 1, &h_mats_res[i * MAX_MATRICULA], h_tiempos_res[i]);
        }
    }
    else {
        std::cout << "No se han encontrado vuelos que cumplan el umbral." << std::endl;
    }

    //Liberamos la memoria de la GPU
    cudaFree(d_arr_delay_ent);
    cudaFree(d_arr_delay_sal);
    cudaFree(d_tail_num_ent);
    cudaFree(d_tail_num_sal);

    std::cout << "[GPU] Analisis finalizado." << std::endl;
}

void cargarFase03(std::vector<float>& retrasos, int operacion_elegida, int variante_elegida) {
    //Calculamos el tamaño del vector de entrada
    int num_vuelos = retrasos.size();
    //Creamos los punteros que usaremos para guardar memoria en la gpu
    float* d_datos;
    int* d_resultado;
    //Variable donde pasaremos el resultado de la gpu
    int h_resultado;

    //Guardamos el espacio necesario para el vector y para la variable resultado en GPU
    cudaMalloc(&d_datos, num_vuelos * sizeof(float));
    cudaMalloc(&d_resultado, sizeof(int));
    //Copiamos los datos del vector de entrada
    cudaMemcpy(d_datos, retrasos.data(), num_vuelos * sizeof(float), cudaMemcpyHostToDevice);
    //Inicializamos el valor inicial de la variable resultado
    //Si es un minimo pondremos un numero grande si es un maximo un numero pequeño
    int valor_inicial = (operacion_elegida == OP_MAX) ? -999999 : 999999;
    //Lo copiamos a la variable de la GPU
    cudaMemcpy(d_resultado, &valor_inicial, sizeof(int), cudaMemcpyHostToDevice);
    //Calculamos el numero de hilos y bloques
    int bloques, hilosPorBloque;
    configurarGridEstandar(num_vuelos, bloques, hilosPorBloque);
    //Calculamos los bytes necesarios para la memoria compartida de los bloques (se usa en parte2,3,4)
    int bytesCompartida = hilosPorBloque * sizeof(int);
    switch (variante_elegida) {
        case VAR_SIMPLE:
            fase03_simple <<< bloques, hilosPorBloque >>> (d_datos, d_resultado, num_vuelos, operacion_elegida);
            break;

        case VAR_BASICA:
            //Ponemos de tercer parámetro la cantidad de memoria compartida que necesitamos en cada bloque en los <<< >>>
            fase03_basica <<< bloques, hilosPorBloque, bytesCompartida >>> (d_datos, d_resultado, num_vuelos, operacion_elegida);
            
            break;

        case VAR_INTERMEDIA:
            //Seguimos trayendonos todos los datos a memoria compartida pero solo un hilo manda el max o min
            fase03_intermedia <<< bloques, hilosPorBloque, bytesCompartida >>>(d_datos, d_resultado, num_vuelos, operacion_elegida);
            break;

        case VAR_PATRON: {
            //Cogemos 512 para que podamos dividir entre 2 y no haya fallo (potencia de 2)
            int threadsPatron = 512;
            int blocksPatron = (num_vuelos + threadsPatron - 1) / threadsPatron;
            int bytesPatron = threadsPatron * sizeof(int);
            fase03_patron << < blocksPatron, threadsPatron, bytesPatron >> > (d_datos, d_resultado, num_vuelos, operacion_elegida);
            break;
        }
        }
    //Esperamos a todos los hilos
    cudaDeviceSynchronize();
    //Copiamos del resultado de la GPU a CPU
    cudaMemcpy(&h_resultado, d_resultado, sizeof(int), cudaMemcpyDeviceToHost);
    std::string nombre_op = (operacion_elegida == OP_MAX) ? "MAXIMO" : "MINIMO";
    std::cout << "\n[GPU] El valor " << nombre_op << " encontrado es: " << h_resultado << " minutos." << std::endl;

    //Liberamos la memoria
    cudaFree(d_datos);
    cudaFree(d_resultado);

    std::cout << "[GPU] Fase 03 finalizada y memoria liberada." << std::endl;
}

void cargarFase04(std::vector<int>& vectorElegido, int umbral_hist, std::unordered_map<int, std::string>& mapa) {
    //Calculamos el numero de vuelos total en el vector
    int num_vuelos = vectorElegido.size();
    if (num_vuelos == 0) return;
    //Calculamos el id maximo entre todos los vuelos para saber cuanta memoria reservar
    int max_id = 0;
    for (int id : vectorElegido) {
        if (id > max_id) max_id = id;
    }
    int tam_histograma = max_id + 1;
    //Creamos los punteros para reservar el espacio
    int* d_ids;
    int* d_histograma;
    //Reservamos el espacio
    cudaMalloc(&d_ids, num_vuelos * sizeof(int));
    cudaMalloc(&d_histograma, tam_histograma * sizeof(int));
    //Inicializamos el histograma a 0 para eliminar valores basura
    cudaMemset(d_histograma, 0, tam_histograma * sizeof(int));
    //Copiamos los datos del vecrtor elegido por el usario a la gpu
    cudaMemcpy(d_ids, vectorElegido.data(), num_vuelos * sizeof(int), cudaMemcpyHostToDevice);
    //Calculamos el numero de hilos y bloques
    int bloques, hilosPorBloque;
    configurarGridEstandar(num_vuelos, bloques, hilosPorBloque);
    //Ejecutamos la funcion de la fase4
    std::cout << "[GPU] Generando histograma con " << tam_histograma << " posibles aeropuertos..." << std::endl;
    fase04 <<< bloques, hilosPorBloque >>> (d_ids, d_histograma, num_vuelos, tam_histograma);
    //Esperamos a todos los hilos
    cudaDeviceSynchronize();
    //Copiamos los resultados
    std::vector<int> h_histograma(tam_histograma);
    cudaMemcpy(h_histograma.data(), d_histograma, tam_histograma * sizeof(int), cudaMemcpyDeviceToHost);
    //Mostramos los resultados
    std::cout << "\n--- RESULTADOS DEL HISTOGRAMA (Umbral: " << umbral_hist << ") ---" << std::endl;
    int unicos = 0;
    for (int i = 0; i < tam_histograma; i++) {
        if (h_histograma[i] >= umbral_hist) {
            unicos++;
            // Buscamos el nombre en el mapa, si no está ponemos el ID
            std::string nombre = (mapa.count(i)) ? mapa[i] : "ID:" + std::to_string(i);

            // Visualización visual (una barra por cada 1000 ocurrencias, por ejemplo)
            printf("%-6s [%-7d]: ", nombre.c_str(), h_histograma[i]);

            int num_barras = h_histograma[i] / 2000; // Ajusta este divisor según el tamaño de tu dataset
            for (int b = 0; b < num_barras && b < 40; b++) std::cout << "I";
            std::cout << std::endl;
        }
    }

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Aeropuertos unicos que superan el umbral: " << unicos << std::endl;

    //Limpieza
    cudaFree(d_ids);
    cudaFree(d_histograma);
}

// Función genérica para configurar los hilos y bloques
void configurarGridEstandar(int n, int& blocks, int& threads) {
    //Buscamos el número de hilos que permite el ordenador, para que el programa pueda ser escalable
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    threads = prop.maxThreadsPerBlock; 
    //Calculamos el número de bloques que necesitamos con el máximo de hilos para abordar el vector completo
    blocks = (n + threads - 1) / threads;
}

void resetYCargar(std::string ruta,
    std::vector<float>& v1, std::vector<float>& v2,
    std::vector<std::string>& v3, std::vector<float>& v4,
    std::vector<int>& v5, std::vector<int>& v6,
    std::unordered_map<int, std::string>& mapa,
    bool& estado) {

    // Limpiamos cualquier rastro de una carga anterior
    v1.clear(); v2.clear(); v3.clear(); v4.clear(); v5.clear(); v6.clear();
    mapa.clear();

    std::cout << "\n Cargando datos desde: " << ruta << "..." << std::endl;
    cargarDataset(ruta, v1, v2, v3, v4, v5, v6, mapa);

    if (v1.empty()) {
        std::cout << "[ERROR] La carga fallo. Los vectores estan vacios." << std::endl;
        estado = false;
    }
    else {
        std::cout << "[OK] Dataset listo con " << v1.size() << " registros." << std::endl;
        estado = true;
    }
}