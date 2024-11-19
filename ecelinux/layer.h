//===========================================================================
// layer.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef LAYER_H
#define LAYER_H

#include "model.h"
#include "typedefs.h"

template <int IN_SIZE, int OUT_SIZE>
void dense(data_t input[IN_SIZE], data_t output[OUT_SIZE], const data_t weight[IN_SIZE][OUT_SIZE], const data_t bias[OUT_SIZE])
{
  for (int i = 0; i < OUT_SIZE; i++)
  {
    data_t sum = bias[i];
    for (int j = 0; j < IN_SIZE; j++)
    {
      sum += input[j] * weight[j][i];
    }
    output[i] = sum;
  }
}

template <int SIZE>
void relu(data_t data[SIZE])
{
  for (int i = 0; i < SIZE; i++)
  {
    if (data[i] < 0)
    {
      data[i] = 0;
    }
  }
}

template <int SIZE>
void apply_dropout(data_t input[SIZE], data_t output[SIZE], const bool mask[SIZE])
{
  for (int i = 0; i < SIZE; i++)
  {
    output[i] = mask[i] ? input[i] : (data_t)0;
  }
}

data_t calculate_mean(data_t outputs[NUM_MONTE_CARLO_RUNS])
{
  data_t sum = 0;
  for (int i = 0; i < NUM_MONTE_CARLO_RUNS; i++)
  {
    sum += outputs[i];
  }
  return (sum / NUM_MONTE_CARLO_RUNS);
}

data_t calculate_variance(data_t outputs[NUM_MONTE_CARLO_RUNS], data_t mean)
{
  data_t var = 0;
  for (int i = 0; i < NUM_MONTE_CARLO_RUNS; i++)
  {
    var += ((outputs[i] - mean) * (outputs[i] - mean));
  }
  return (var / (NUM_MONTE_CARLO_RUNS));
}

template <int ITERATIONS, int NEURONS>
void generate_binary_matrix(bit matrix[ITERATIONS][NEURONS], float zero_percentage)
{
  int total_zeros = (int)(zero_percentage * ITERATIONS * NEURONS);
  int random_seed = 123; // Seed for LFSR
  int zeros_added = 0;

  // LFSR-based pseudo-random number generation
  ap_uint<16> lfsr = random_seed;

  for (int i = 0; i < ITERATIONS; i++)
  {
    for (int j = 0; j < NEURONS; j++)
    {
      // Update LFSR
      bool bit = lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[5];
      lfsr = (lfsr >> 1) | (bit << 15);

      if (zeros_added < total_zeros && (lfsr & 0x1))
      {
        matrix[i][j] = 0;
        zeros_added++;
      }
      else
      {
        matrix[i][j] = 1;
      }
    }
  }
}
#endif
