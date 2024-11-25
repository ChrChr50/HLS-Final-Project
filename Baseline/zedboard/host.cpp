#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>

#include <iostream>
#include <fstream>

#include "typedefs.h"
#include "timer.h"
#include "model.h"
#include "mlp.h"

const int TEST_SIZE = 150;  // Number of rows in x_test
const int FEATURE_COUNT = 9; // Number of features (columns) per row

void read_test_images(int8_t test_images[TEST_SIZE][256]) {
  std::ifstream infile("data/test_images.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      for (int pixel = 0; pixel < 256; pixel++) {
        int i;
        infile >> i;
        test_images[index][pixel] = i;
      }
    }
    infile.close();
  }
}

void read_test_dataset(data_t x_test[TEST_SIZE][FEATURE_COUNT]) {
    std::ifstream infile("test_data.cpp");
    
    // Ensure the file opens successfully
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file." << std::endl;
        return;
    }

    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < FEATURE_COUNT; j++) {
            int temp;
            infile >> temp;  // Read a floating-point value
            x_test[i][j] = (data_t)temp;  // Convert to the FPGA-compatible data type
        }
    }

    infile.close();
}

void read_test_dataset(float x_test[TEST_SIZE][FEATURE_COUNT], const std::string& file_name) {
    std::ifstream infile(file_name);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return;
    }

    std::string line;
    int row = 0;

    while (std::getline(infile, line) && row < TEST_SIZE) {
        std::istringstream ss(line);
        for (int col = 0; col < FEATURE_COUNT; col++) {
            if (!(ss >> x_test[row][col])) {
                std::cerr << "Error: Malformed data on row " << row << ", column " << col << std::endl;
                return;
            }
        }
        row++;
    }

    if (row != TEST_SIZE) {
        std::cerr << "Error: Expected " << TEST_SIZE << " rows, but only read " << row << " rows." << std::endl;
    }

    infile.close();
    std::cout << "Successfully loaded x_test dataset from " << file_name << std::endl;
}

void read_test_labels(int test_labels[TEST_SIZE]) {
  std::ifstream infile("data/test_labels.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      infile >> test_labels[index];
    }
    infile.close();
  }
}

//--------------------------------------
// main function
//--------------------------------------
int main(int argc, char **argv) {
  // Open channels to the FPGA board.
  // These channels appear as files to the Linux OS
  int fdr = open("/dev/xillybus_read_32", O_RDONLY);
  int fdw = open("/dev/xillybus_write_32", O_WRONLY);

  // Check that the channels are correctly opened
  if ((fdr < 0) || (fdw < 0)) {
    fprintf(stderr, "Failed to open Xillybus device channels\n");
    exit(-1);
  }

  int8_t test_images[TEST_SIZE][256];
  bit32_t test_image;
  int test_labels[TEST_SIZE];

  // Timer
  Timer timer("MLP FPGA");
  int nbytes;
  int error = 0;
  int num_test_insts = 0;
  bit32_t interpreted_digit;
  float correct = 0.0;

  read_test_images(test_images);
  read_test_labels(test_labels);

  //--------------------------------------------------------------------
  // Run it once without timer to test accuracy
  //--------------------------------------------------------------------
  std::cout << "Testing accuracy over " << TEST_SIZE << " images." << std::endl;
  // Send data to accelerator
  for (int i = 0; i < TEST_SIZE; ++i) {
    // Send 32-bit value through the write channel
    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < 32; k++) {
        test_image(k, k) = test_images[i][j * 32 + k];
      }
      nbytes = write(fdw, (void *)&test_image, sizeof(test_image));
      assert(nbytes == sizeof(test_image));
    }
  }
  // Receive data from the accelerator
  for (int i = 0; i < TEST_SIZE; ++i) {
    bit32_t output;
    nbytes = read(fdr, (void *)&output, sizeof(output));
    assert(nbytes == sizeof(output));
    // verify results
    if (output == test_labels[i])
      correct += 1.0;
  }
  // Calculate error rate
  std::cout << "Accuracy: " << correct / TEST_SIZE << std::endl;

  //--------------------------------------------------------------------
  // Run it 20 times to test performance
  //--------------------------------------------------------------------
  std::cout << "Testing performance over " << REPS*TEST_SIZE << " images." << std::endl;
  timer.start();
  // Send data to accelerator
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < TEST_SIZE; ++i) {
      // Send 32-bit value through the write channel
      for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 32; k++) {
          test_image(k, k) = test_images[i][j * 32 + k];
        }
        nbytes = write(fdw, (void *)&test_image, sizeof(test_image));
        assert(nbytes == sizeof(test_image));
      }
    }
  }
  // Receive data from the accelerator
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < TEST_SIZE; ++i) {
      bit32_t output;
      nbytes = read(fdr, (void *)&output, sizeof(output));
      assert(nbytes == sizeof(output));
      // verify results
      if (output == test_labels[i])
        correct += 1.0;
    }
  }
  timer.stop();
  // total time wil be automatically printed upon exit.

  return 0;
}
