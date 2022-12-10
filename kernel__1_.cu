#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

using namespace std;
#include "AudioFile.h"

#define PROJECT_BINARY_DIR "C:/Users/mlaramie24/source/repos/AudioWork"

AudioFile<float> loadAudio(vector<float>& v) {
    //---------------------------------------------------------------
    // 1. Set a file path to an audio file on your machine

    const std::string inputFilePath = std::string(PROJECT_BINARY_DIR) + "/AudioWork/examples_test-audio.wav";

    //---------------------------------------------------------------
    // 2. Create an AudioFile object and load the audio file

    AudioFile<float> a;
    bool loadedOK = a.load(inputFilePath);

    //---------------------------------------------------------------
    // 3. Let's apply a gain to every audio sample

    for (int i = 0; i < a.getNumSamplesPerChannel(); i++) {
        for (int channel = 0; channel < a.getNumChannels(); channel++) {
            v.push_back(a.samples[channel][i]);
        }
    }
    return a;
}

int main() {
    std::cout << "**********************" << std::endl;
    std::cout << "funking arounddd" << std::endl;
    std::cout << "**********************" << std::endl << std::endl;


    //---------------------------------------------------------------
    // Step 1: Input Data
    vector<float> audio;
    AudioFile<float> a = loadAudio(audio);
    int bigness = audio.size();

    cufftReal* hostInputData = (cufftReal*)malloc(bigness * sizeof(cufftReal));
    for (int i = 0; i < bigness; i++) {
        hostInputData[i] = (cufftReal)audio[i];
    }

    //---------------------------------------------------------------
    // Step 2: Device memory allocation
    cufftComplex* deviceInputData;
    cudaMalloc((void**)&deviceInputData, bigness * sizeof(cufftComplex));
    cudaMemcpy(deviceInputData, hostInputData, bigness * sizeof(cufftReal), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------
    // Step 3: Allocate host & device memory for output data collection
    cufftComplex* hostOutputData = (cufftComplex*)malloc((bigness / 2 + 1) * 1 * sizeof(cufftComplex));

    cufftComplex* deviceOutputData;
    cudaMalloc((void**)&deviceOutputData, (bigness / 2 + 1) * sizeof(cufftComplex));
    
    // Setup cufft plan and handle
    cufftHandle handle;
    cufftResult cufftStatus = cufftPlan1d(&handle, bigness, CUFFT_R2C, 1);

    //---------------------------------------------------------------
    // Step 4: Execute R2C FFT
    cufftStatus = cufftExecR2C(handle, (cufftReal*)deviceInputData, deviceOutputData);

    //---------------------------------------------------------------
    // Step 5: Transfer results from Device -> Host
    cudaMemcpy(hostOutputData, deviceOutputData, (bigness / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < (bigness / 2 + 1); i++) {
        printf("%i %f %f\n", i, hostOutputData[i].x, hostOutputData[i].y);
    }

    // Cleanup
    cufftDestroy(handle);

    //std::string outputFilePath = "result_audio.wav"; // change this to somewhere useful for you
    //a.save(outputFilePath, AudioFileFormat::Aiff);
    //cout << audio.size() << endl;
}