#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
using namespace std;
#include "AudioFile.h"

#define PROJECT_BINARY_DIR "C:/Users/lcalderon-aceituno24/source/repos/test_audio/test_audio"

void d_doNothing();

int main() {
    //---------------------------------------------------------------
    std::cout << "**********************" << std::endl;
    std::cout << "funking arounddd" << std::endl;
    std::cout << "**********************" << std::endl << std::endl;

    d_doNothing();

    //---------------------------------------------------------------
    // 1. Set a file path to an audio file on your machine
    const std::string inputFilePath = std::string(PROJECT_BINARY_DIR) + "/examples_test-audio.wav";

    //---------------------------------------------------------------
    // 2. Create an AudioFile object and load the audio file

    AudioFile<float> a;
    vector<float> original; 
    vector<float> amplified;
    bool loadedOK = a.load(inputFilePath);
    // run diagnostics on a 
    std::cout << "Bit Depth: " << a.getBitDepth() << std::endl;
    std::cout << "Sample Rate: " << a.getSampleRate() << std::endl;
    std::cout << "Num Channels: " << a.getNumChannels() << std::endl;


    /** If you hit this assert then the file path above
     probably doesn't refer to a valid audio file */
    assert(loadedOK);

    //---------------------------------------------------------------
    // 3. Let's apply a gain to every audio sample

    float gain = 5.0f;

    int toggle = 0;
    for (int i = 0; i < a.getNumSamplesPerChannel(); i++)
    {
        for (int channel = 0; channel < a.getNumChannels(); channel++)
        {
            original.push_back(a.samples[channel][i]); // copy original signal 
            a.samples[channel][i] = a.samples[channel][i] * gain;
            amplified.push_back(a.samples[channel][i]); // copy amplified signal 
        }
    }

    //---------------------------------------------------------------
    // 4. Write audio file to disk

    //std::string outputFilePath = "squashed2.wav"; // change this to somewhere useful for you
    //a.save(outputFilePath, AudioFileFormat::Aiff);
}