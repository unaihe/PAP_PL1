#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>

__constant__ int d_umbral;

void cargarDataset(std::string ruta,
    std::vector<float>& dep_delay,
    std::vector<float>& arr_delay,
    std::vector<std::string>& tail_num, 
    std::vector<float>& weather_delay,
    std::vector<int>& origin_id,
    std::vector<int>& dest_id);

void cargarFase01(std::vector<float>& retrasos, int umbral);

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

    //Ruta al csv con los datos
    std::string ruta_csv;
    //Opcion para insertar ruta por teclado o elegir la generica
    std::cout << "Introduzca la ruta base del dataset (Enter para ruta por defecto): ";
    std::getline(std::cin, ruta_csv);
    //Si ruta_csv es vacio, se usa la ruta por defecto 
    if (ruta_csv.empty()) {
        ruta_csv = "C:\\Users\\Unai\\OneDrive\\Documentos\\Airline_dataset.csv"; 
    }

    //Leer el CSV de datos
    std::cout << "Inicio de lectura de datos" << std::endl;
    cargarDataset(ruta_csv, h_dep_delay, h_arr_delay, h_tail_num, h_weather_delay, h_origin_id, h_dest_id);

    char opcion;
    do {
        std::cout << "\nMenu de opciones:" << std::endl;
        std::cout << "(1) Retraso en salida" << std::endl;
        std::cout << "(2) Retraso en llegada" << std::endl;
        std::cout << "(3) Reduccion de retraso" << std::endl;
        std::cout << "(4) Histograma de aeropuertos" << std::endl;
        std::cout << "(x) Salir" << std::endl;
        std::cout << "Seleccione una opcion: ";
        std::cin >> opcion;

        switch (opcion) {
        case '1': {
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
        case '2': {
            // Retraso en aterrizajes 
            break;
        }
        case '3': {
            // Reducción (Máximos/Mínimos) 
            break;
        }
        case '4': {
            // Histograma 
            break;
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
    std::vector<int>& dest_id) {

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

    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        std::string celda;
        int columna = 0;

        while (std::getline(ss, celda, ',')) {
            // Columna 3: Matrícula (TAIL_NUM) para Fase 02 
            if (columna == 3) {
                tail_num.push_back(celda);
            }
            // Columna 5: ID Aeropuerto Origen (ORIGIN_SEQ_ID) para Fase 04 
            else if (columna == 5) {
                if (celda.empty()) origin_id.push_back(0);
                else origin_id.push_back(std::stoi(celda));
            }
            // Columna 7: ID Aeropuerto Destino (DEST_SEQ_ID) para Fase 04 
            else if (columna == 7) {
                if (celda.empty()) dest_id.push_back(0);
                else dest_id.push_back(std::stoi(celda));
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
    // Traza final requerida [cite: 29]
    std::cout << "Carga finalizada con exito." << std::endl;
}

void cargarFase01(std::vector<float>& retrasos, int umbral) {

    //Calculamos el numero de vuelos del vector para poder guardar el espacio necesario en memoria
    int num_vuelos = retrasos.size();
    //Creamos una variable puntero donde se iniciará el vector en la GPU
    float* d_retrasos;
    
    //Buscamos el número de hilos que permite el ordenador, para que el programa pueda ser escalable
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int hilosPorBloque = prop.maxThreadsPerBlock;
    //Calculamos el número de bloques que necesitamos con el máximo de hilos para abordar el vector completo
    int bloques = (num_vuelos + hilosPorBloque - 1) / hilosPorBloque;

    std::cout << "[GPU] Procesando " << num_vuelos << " vuelos..." << std::endl;

    //Guardamos la memoria necesaria en la gpu a través del tamaño del vector calculado con el .size
    cudaMalloc(&d_retrasos, num_vuelos * sizeof(float));
    //Copiamos a la memoria de la GPU el vector entero de retrasos
    cudaMemcpy(d_retrasos, retrasos.data(), num_vuelos * sizeof(float), cudaMemcpyHostToDevice);

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