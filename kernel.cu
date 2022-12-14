//---------------------------------------------------------------
// AUDIO SLICING FINAL PROJECT
// CS 315 Distributed Scalable Computing
// Dr. Qian Mao 
// Whitworth University
// DEVELOPED BY:
// LYDIA CALDERON-ACEITUNO, MICHAEL LARAMIE, OWEN FOSTER
//---------------------------------------------------------------
// USING LIBRARY DEVELOPED BY ADAM STARK
// https://github.com/adamstark/AudioFile
// https://www.adamstark.co.uk
//---------------------------------------------------------------

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <algorithm>
#include <cmath>

using namespace std;
#include "AudioFile.h"

#define PROJECT_BINARY_DIR "C:/Users/mlaramie24/source/repos/AudioWork/AudioWork"

// Returns an AudioFile object, also fills a given vector with the 1d conversion of the audio file.
AudioFile<float> loadAudio(vector<float>& v) {
    //---------------------------------------------------------------
    // 1. Set a file path to the audio file

    const std::string inputFilePath = std::string(PROJECT_BINARY_DIR) + "/sample_3.wav";

    //---------------------------------------------------------------
    // 2. Load the audio file

    AudioFile<float> a;
    bool loadedOK = a.load(inputFilePath);

    printf("%d %d\n", a.getNumChannels(), a.getNumSamplesPerChannel()); // print stats for testing

    //---------------------------------------------------------------
    // Convert data from Multi-Dimensional Array -> One-Dimensional Array
    for (int i = 0; i < a.getNumSamplesPerChannel(); i++) {
        v.push_back((a.samples[0][i] + a.samples[1][i]) / 2);
    }
    return a;
}

// Function to slice the complex data into two lists
// Takes in the index of the highest magnitude frequency
// Slices 100 indices below and above
void findBins(int index, cufftComplex* hod, cufftComplex* crazy, int size) {
    // Grab all complex data points other than the slice
    for (int i = 0; i < size / 2 + 1; i++) {
        if (i > index - 100 && i < index + 100) {
            crazy[i].x = hod[i].x;
            crazy[i].y = hod[i].y;
        }
        else {
            crazy[i].x = 0.0f;
            crazy[i].y = 0.0f;
        }
    }
    // Grab the slice
    for (int i = index - 100; i < index + 100; i++) {
        hod[i].x = 0.0f;
        hod[i].y = 0.0f;
    }
    cout << "INDEX: " << index << endl; // print index for testing
}

//---------------------------------------------------------------
// Main
int main() {
    // Print basic info for testing in the console
    std::cout << "**********************" << std::endl;
    std::cout << "cuFFT audio transform" << std::endl;
    std::cout << "**********************" << std::endl << std::endl;


    //---------------------------------------------------------------
    // Step 1: Gather Input Data
    vector<float> audio;
    AudioFile<float> a = loadAudio(audio);
    int size = audio.size();

    // Put all data points from audio file into a cufftReal* object 
    // to use in FFT
    cufftReal* hostInputData = (cufftReal*)malloc(size * sizeof(cufftReal));
    for (int i = 0; i < size; i++) {
        hostInputData[i] = (cufftReal)audio[i];
    }

    // Print out sample data to test
    cout << "REAL DATA BT: " << endl;
    for (int i = 0; i < 30; i++) {
        printf("%i %f\n", i, hostInputData[i]);
    }

    //---------------------------------------------------------------
    // Step 2: Device memory allocation
    cufftReal* deviceInputData;
    cudaMalloc((void**)&deviceInputData, size * sizeof(cufftReal));
    cudaMemcpy(deviceInputData, hostInputData, size * sizeof(cufftReal), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------
    // Step 3: Allocate host & device memory for output data collection
    cufftComplex* hostOutputData = (cufftComplex*)malloc((size / 2 + 1) * 1 * sizeof(cufftComplex));

    cufftComplex* deviceOutputData;
    cudaMalloc((void**)&deviceOutputData, (size / 2 + 1) * sizeof(cufftComplex));

    // Setup cufft plan and handle
    cufftHandle handle;
    cufftResult cufftStatus = cufftPlan1d(&handle, size, CUFFT_R2C, 1);

    //---------------------------------------------------------------
    // Step 4: Execute R2C FFT
    cufftStatus = cufftExecR2C(handle, (cufftReal*)deviceInputData, deviceOutputData);

    //---------------------------------------------------------------
    // Step 5: Transfer results from Device -> Host
    cudaMemcpy(hostOutputData, deviceOutputData, (size / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    //---------------------------------------------------------------
    // Create vector of magnitudes of each complex data point
    // vector has length of (size / 2) + 1
    vector<float> vecto;
    for (int i = 0; i < size / 2 + 1; i++) {
        //printf("%f %f\n", hostOutputData[i].x, hostOutputData[i].y);
        float mag = sqrt((hostOutputData[i].x * hostOutputData[i].x) + (hostOutputData[i].y * hostOutputData[i].y));
        vecto.push_back(mag);
    }

    // Clone complex data in order to have two copies to slice in
    cufftComplex* crazee = (cufftComplex*)malloc((size / 2 + 1) * sizeof(cufftComplex));
    for (int i = 0; i < size / 2 + 1; i++) {
        crazee[i].x = hostOutputData[i].x;
        crazee[i].y = hostOutputData[i].y;
    }

    // Find the index with the highest magnitude
    // this is equivalent to the loudest frequency
    float highest = *max_element(vecto.begin(), vecto.end());

    // If you want to generate a CSV file
    // CSV file needed for graphing function 
    std::ofstream ffout("jingle_out.csv");
    ffout << "\"Re\"" << "," << "\"Im\"" << std::endl;
    for (int i = 0; i < (size / 2 + 1); i++)
    {
        ffout << hostOutputData[i].x << "," << hostOutputData[i].y << std::endl;
    }
    ffout.close();

    //---------------------------------------------------------------
    // Iterate through vector of magnitudes to find the index of the highest magnitude
    // once highest magnitude index is found, call findBins(...) to slice the data around the highest index
    for (int i = 0; i < size / 2 + 1; i++) {
        if (vecto[i] == highest) {
            findBins(i, hostOutputData, crazee, size);
        }
    }

    //---------------------------------------------------------------
    // Begin complex to real inverse fourier transform to convert the now sliced complex data into 
    // real data to write to audio file
    //---------------------------------------------------------------
    // Surrounding data conversion back to real data
    // Step 1: Gather Input Data
    cufftComplex* hostInputData2 = (cufftComplex*)malloc((size / 2 + 1) * sizeof(cufftComplex));
    for (int i = 0; i < size / 2 + 1; i++) {
        hostInputData2[i].x = hostOutputData[i].x;
        hostInputData2[i].y = hostOutputData[i].y;
    }

    // Step 2: Device memory Allocation
    cufftComplex* deviceInputData2;
    cudaMalloc((void**)&deviceInputData2, (size / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(deviceInputData2, hostInputData2, (size / 2 + 1) * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------
    // Step 3: Allocate host & device memory for output data collection
    cufftReal* hostOutputData2 = (cufftReal*)malloc(size * sizeof(cufftReal));
    cufftReal* deviceOutputData2;
    cudaMalloc((void**)&deviceOutputData2, size * sizeof(cufftReal));

    // Setup cufft plan and handle
    cufftHandle handle2;
    cufftResult cufftStatus2 = cufftPlan1d(&handle2, (size / 2 + 1), CUFFT_C2R, 1);

    //---------------------------------------------------------------
    // Step 4: Execute C2R FFT
    cufftStatus2 = cufftExecC2R(handle2, (cufftComplex*)deviceInputData2, deviceOutputData2);

    //---------------------------------------------------------------
    // Step 5: Transfer results from Device -> Host
    cudaMemcpy(hostOutputData2, deviceOutputData2, size * sizeof(cufftReal), cudaMemcpyDeviceToHost);

    //---------------------------------------------------------------
    // Sliced data conversion back to real data
    // Step 1: Gather input Data
    cufftComplex* hostInputData3 = (cufftComplex*)malloc((size / 2 + 1) * sizeof(cufftComplex));
    for (int i = 0; i < size / 2 + 1; i++) {
        hostInputData3[i].x = crazee[i].x;
        hostInputData3[i].y = crazee[i].y;
    }

    // Step 2: Device memory Allocation
    cufftComplex* deviceInputData3;
    cudaMalloc((void**)&deviceInputData3, (size / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(deviceInputData3, hostInputData3, (size / 2 + 1) * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------
    // Step 3: Allocate host & device memory for output data collection
    cufftReal* hostOutputData3 = (cufftReal*)malloc(size * sizeof(cufftReal));
    cufftReal* deviceOutputData3;
    cudaMalloc((void**)&deviceOutputData3, size * sizeof(cufftReal));

    // Setup cufft plan and handle
    cufftHandle handle3;
    cufftResult cufftStatus3 = cufftPlan1d(&handle3, (size / 2 + 1), CUFFT_C2R, 1);

    //---------------------------------------------------------------
    // Step 4: Execute C2R FFT
    cufftStatus3 = cufftExecC2R(handle3, (cufftComplex*)deviceInputData3, deviceOutputData3);

    //---------------------------------------------------------------
    // Step 5: Transfer results from Device -> Host
    cudaMemcpy(hostOutputData3, deviceOutputData3, size * sizeof(cufftReal), cudaMemcpyDeviceToHost);

    //---------------------------------------------------------------
    // Cleanup
    cufftDestroy(handle);
    cufftDestroy(handle2);
    cufftDestroy(handle3);
    //---------------------------------------------------------------

    // Write both selections of data to unique audio files
    // Surrounding data
    AudioFile<float> output;
    string outputFilePath = "result_audio2.wav"; // change this to somewhere useful for you
    cout << outputFilePath << endl;

    for (int j = 0; j < size; j++) {
        output.samples[0].push_back(hostOutputData2[j] / size);
        output.samples[0].push_back(0);
    }
    output.save(outputFilePath, AudioFileFormat::Wave);
        cout << audio.size() << endl;

    // Sliced data
    AudioFile<float> output2;
    string outputFilePath2 = "result_audio3.wav"; // change this to somewhere useful for you
    cout << outputFilePath2 << endl;

    for (int j = 0; j < size; j++) {
        output2.samples[0].push_back(hostOutputData3[j] / size);
        output2.samples[0].push_back(0);
    }
    output2.save(outputFilePath2, AudioFileFormat::Wave);
    cout << audio.size() << endl;
}
