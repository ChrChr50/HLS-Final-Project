-- ==============================================================
-- Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.2 (64-bit)
-- Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity mlp_monte_carlo_wg8j_rom is 
    generic(
             DWIDTH     : integer := 13; 
             AWIDTH     : integer := 9; 
             MEM_SIZE    : integer := 512
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of mlp_monte_carlo_wg8j_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "1001101011001", 1 => "1001100010000", 2 => "0000010100110", 
    3 => "1111110111101", 4 => "1011000101001", 5 => "0000001100001", 
    6 => "1011000111011", 7 => "1010001000101", 8 => "1010010001011", 
    9 => "1111011100110", 10 => "1010111011100", 11 => "0000010011011", 
    12 => "1011000101000", 13 => "1010000100111", 14 => "0010111011010", 
    15 => "1010110100010", 16 => "1111010100110", 17 => "0000100010101", 
    18 => "0000010111110", 19 => "0000100101111", 20 => "1111100000010", 
    21 => "0000101011100", 22 => "1111010110010", 23 => "1111100010100", 
    24 => "1111101011100", 25 => "0000101000101", 26 => "0000010011000", 
    27 => "1111100011011", 28 => "1111100110111", 29 => "0000010011101", 
    30 => "1111101111100", 31 => "0000011100010", 32 => "0000011000101", 
    33 => "0011010111011", 34 => "1111111000100", 35 => "1111001010001", 
    36 => "0010101101101", 37 => "1111100011111", 38 => "0011100011001", 
    39 => "0011110010000", 40 => "0011001001100", 41 => "1111100111000", 
    42 => "0011010111101", 43 => "0000000110111", 44 => "0011011001100", 
    45 => "0011100111101", 46 => "1110110100100", 47 => "0011010111001", 
    48 => "0001100001001", 49 => "0000011000001", 50 => "0000011110011", 
    51 => "1111100001111", 52 => "0000100111100", 53 => "0000100001010", 
    54 => "0000001001111", 55 => "0000011000111", 56 => "0001000000010", 
    57 => "1111011100010", 58 => "0000110010111", 59 => "1111011101000", 
    60 => "0000111010001", 61 => "1111111000001", 62 => "1101101010011", 
    63 => "1111111110101", 64 => "1111111101001", 65 => "0001000010110", 
    66 => "1111101010011", 67 => "1111001011111", 68 => "0010000010100", 
    69 => "1111111010100", 70 => "0001110101011", 71 => "0010000110111", 
    72 => "0001111110011", 73 => "0000000110000", 74 => "0000110100100", 
    75 => "1111110001101", 76 => "0001110100110", 77 => "0000110101100", 
    78 => "0000100101011", 79 => "0001111001111", 80 => "1111111000111", 
    81 => "0001001111111", 82 => "1111111100110", 83 => "0000001000011", 
    84 => "0000001111010", 85 => "1111101011110", 86 => "0000011100111", 
    87 => "0001001011001", 88 => "0001001100010", 89 => "1111010010001", 
    90 => "0000001101111", 91 => "0000010111010", 92 => "0000110111111", 
    93 => "0000111110111", 94 => "1111101100101", 95 => "0001100011001", 
    96 => "0000001110110", 97 => "1111101010001", 98 => "0000001100001", 
    99 => "1111010100100", 100 => "1111011111011", 101 => "0000001000010", 
    102 => "1111110100001", 103 => "0000010110011", 104 => "1111101110000", 
    105 => "1111100110111", 106 => "0000001010101", 107 => "0000011011010", 
    108 => "1111100110101", 109 => "0000100111011", 110 => "1111111011100", 
    111 => "1111111111001", 112 => "1111100000011", 113 => "0000010101101", 
    114 => "0000011101110", 115 => "1111100010010", 116 => "0000111110000", 
    117 => "1111100011011", 118 => "0000000011110", 119 => "0001011101111", 
    120 => "0001011101101", 121 => "1111101000110", 122 => "0001001111010", 
    123 => "1111000101011", 124 => "0000111100000", 125 => "0000001011100", 
    126 => "1111110110000", 127 => "0000010101111", 128 => "1111100101100", 
    129 => "1111100000100", 130 => "0000100001101", 131 => "0000001001010", 
    132 => "0000001100110", 133 => "1111011111010", 134 => "1111010110111", 
    135 => "0000101000010", 136 => "0000011011110", 137 => "1111110100000", 
    138 => "1111011011101", 139 => "0000000101001", 140 => "0000000110011", 
    141 => "0000011110101", 142 => "0000101100100", 143 => "1111010010111", 
    144 => "1110110111001", 145 => "0000010100100", 146 => "1111111111101", 
    147 => "1111100110110", 148 => "1111101101101", 149 => "0000011010111", 
    150 => "1111111100011", 151 => "0000110001000", 152 => "0000111000101", 
    153 => "1111101110000", 154 => "0000001111101", 155 => "1111011101111", 
    156 => "1111111010110", 157 => "0000101010000", 158 => "1111101111100", 
    159 => "0000010101111", 160 => "1101110100100", 161 => "0001001111111", 
    162 => "0000001111000", 163 => "1111010110010", 164 => "0000101001100", 
    165 => "0000001110011", 166 => "0000101110001", 167 => "0000100100001", 
    168 => "0000100111101", 169 => "0000011000010", 170 => "0000111001010", 
    171 => "1111110001110", 172 => "0000111001000", 173 => "0000111001110", 
    174 => "1110111011110", 175 => "0000100100110", 176 => "1100011110101", 
    177 => "0000101011011", 178 => "0000011011101", 179 => "1111001001110", 
    180 => "0000110101101", 181 => "0000010011000", 182 => "0000011011010", 
    183 => "0000001010111", 184 => "0000000011100", 185 => "0000010000000", 
    186 => "0000111100001", 187 => "1111111101101", 188 => "0001001000010", 
    189 => "0000001000011", 190 => "1111010100111", 191 => "0000101100101", 
    192 => "0000100100110", 193 => "1111101000100", 194 => "0000101010010", 
    195 => "1111011100001", 196 => "0000000000101", 197 => "0000100111010", 
    198 => "1111100010011", 199 => "1111011101111", 200 => "0000000111111", 
    201 => "1111100111110", 202 => "1111101100001", 203 => "1111111000011", 
    204 => "0000100000010", 205 => "0000011111100", 206 => "1111011100111", 
    207 => "0000001101000", 208 => "1111100110000", 209 => "1111110010111", 
    210 => "0000100100110", 211 => "0000011010001", 212 => "1111110011101", 
    213 => "0000100100100", 214 => "0000100101010", 215 => "1111011111110", 
    216 => "1111100101111", 217 => "1111110110011", 218 => "0000010100001", 
    219 => "0000000100001", 220 => "0000100110101", 221 => "1111101010111", 
    222 => "1111101011001", 223 => "1111100011100", 224 => "1111110010001", 
    225 => "1111111011100", 226 => "1111100010101", 227 => "0000000111011", 
    228 => "1111111100101", 229 => "0000010010011", 230 => "0000111010101", 
    231 => "0001000100110", 232 => "0000101100111", 233 => "0000011011011", 
    234 => "1111110011111", 235 => "1111110111011", 236 => "0001000111101", 
    237 => "0001000001001", 238 => "1101010101101", 239 => "0000110010101", 
    240 => "1111011101100", 241 => "1111101101001", 242 => "1111111010101", 
    243 => "0000100001100", 244 => "1111110111011", 245 => "1111101010100", 
    246 => "0000011111101", 247 => "1111110110000", 248 => "1111101111010", 
    249 => "0000010011110", 250 => "1111100101000", 251 => "0000000001100", 
    252 => "1111110011010", 253 => "0000001001010", 254 => "0000001001110", 
    255 => "1111100001000", 256 => "1101010000100", 257 => "1111101101101", 
    258 => "1111010101100", 259 => "0000010001001", 260 => "1111111001101", 
    261 => "1111011100011", 262 => "1111100101010", 263 => "0000001110011", 
    264 => "1111111011100", 265 => "1111110100100", 266 => "1111110010110", 
    267 => "1111100101101", 268 => "1111111010100", 269 => "0000110000000", 
    270 => "1100111010001", 271 => "0000011101100", 272 => "0000001101110", 
    273 => "0000111001111", 274 => "1111111010101", 275 => "0000011010010", 
    276 => "0000110111001", 277 => "1111011100111", 278 => "0000001100010", 
    279 => "0000100011000", 280 => "0000001011110", 281 => "1111011001000", 
    282 => "0000110110100", 283 => "1111001101001", 284 => "0000101110001", 
    285 => "0000011001100", 286 => "0000000000100", 287 => "1111101111010", 
    288 => "0000001110111", 289 => "0001110011111", 290 => "1111100111101", 
    291 => "1111100110111", 292 => "0001011000100", 293 => "1111101100010", 
    294 => "0001010100110", 295 => "0001010000110", 296 => "0001100101110", 
    297 => "1111010100111", 298 => "0000101111001", 299 => "1111101101000", 
    300 => "0001010101101", 301 => "0001101000100", 302 => "0000110010011", 
    303 => "0001111111100", 304 => "1111110100100", 305 => "0000111011111", 
    306 => "1111100110010", 307 => "1111101110110", 308 => "1111101101110", 
    309 => "0000000100001", 310 => "0000100001001", 311 => "0000001111111", 
    312 => "0000011101110", 313 => "1111001111101", 314 => "1111110111010", 
    315 => "1111111110101", 316 => "1111111110110", 317 => "0000110001011", 
    318 => "0000100101010", 319 => "0000101000000", 320 => "0001110110111", 
    321 => "0001011100001", 322 => "0000001110010", 323 => "0000011101111", 
    324 => "0001000111000", 325 => "1111100100010", 326 => "0000111000000", 
    327 => "0001000100111", 328 => "0001010011001", 329 => "1111111111000", 
    330 => "0000010111111", 331 => "1111011111101", 332 => "0000101001000", 
    333 => "0001001100010", 334 => "1111100100101", 335 => "0001000000100", 
    336 => "1111110111001", 337 => "0001110011001", 338 => "1111110001010", 
    339 => "0000011010110", 340 => "0000100011111", 341 => "0000001100001", 
    342 => "0001000010011", 343 => "0000011001101", 344 => "0000111001101", 
    345 => "0000011001000", 346 => "0001011100111", 347 => "0000001001011", 
    348 => "0000111001101", 349 => "0001010111101", 350 => "1110111101011", 
    351 => "0001110100010", 352 => "0000011101100", 353 => "0000010110101", 
    354 => "1111010100010", 355 => "0000010000111", 356 => "0000000000011", 
    357 => "1111110001001", 358 => "0000111011011", 359 => "0001000010011", 
    360 => "0000101010100", 361 => "1111111001000", 362 => "0000011110111", 
    363 => "0000011001111", 364 => "0001000010100", 365 => "0000011100001", 
    366 => "0001001000110", 367 => "0000001001011", 368 => "0000010101011", 
    369 => "0010100110011", 370 => "1111111101000", 371 => "0000001110010", 
    372 => "0011011100011", 373 => "0000011100100", 374 => "0010001111110", 
    375 => "0011100011101", 376 => "0011010101001", 377 => "0000100011000", 
    378 => "0011001000100", 379 => "1111110000001", 380 => "0011010101110", 
    381 => "0011010111110", 382 => "1010100100101", 383 => "0010111000010", 
    384 => "1110010110111", 385 => "0010001110111", 386 => "0000100110101", 
    387 => "0000001000010", 388 => "0001101000101", 389 => "0000000010110", 
    390 => "0010011010110", 391 => "0010111101011", 392 => "0001100110111", 
    393 => "0000000010111", 394 => "0010000000101", 395 => "1111110010101", 
    396 => "0010111000100", 397 => "0010100101101", 398 => "1110111011010", 
    399 => "0010000011010", 400 => "1100110000110", 401 => "1111101111011", 
    402 => "1111010001001", 403 => "1111001011111", 404 => "0000100000010", 
    405 => "1111011011101", 406 => "0000100000100", 407 => "1111100111110", 
    408 => "0000011110011", 409 => "1111011111100", 410 => "1111111101110", 
    411 to 412=> "1111110011111", 413 => "0000101111110", 414 => "1111001101100", 
    415 => "1111111100011", 416 => "1111100111110", 417 => "0000011000110", 
    418 => "0000011111010", 419 => "0000100111100", 420 => "1111010011101", 
    421 => "1111011010110", 422 => "1111111010111", 423 => "1111111000011", 
    424 => "1111011010101", 425 => "0000000111011", 426 => "0000001101101", 
    427 => "0000010000100", 428 => "1111101011110", 429 => "0000010111010", 
    430 => "0000011101100", 431 => "0000001111110", 432 => "0000011110101", 
    433 => "1111110010100", 434 => "1111111010111", 435 => "1111100111101", 
    436 => "0001000101011", 437 => "1111011011010", 438 => "0000111100111", 
    439 => "0000101101101", 440 => "0001001110011", 441 => "1111010001000", 
    442 => "0000101011010", 443 => "1111111110101", 444 => "0001001101011", 
    445 => "0000101010011", 446 => "0000010000100", 447 => "0000011000101", 
    448 => "0000101111110", 449 => "0001001000101", 450 => "0000001101000", 
    451 => "0000001010111", 452 => "0000111100100", 453 => "0000001100110", 
    454 => "0001001011101", 455 => "0000000010010", 456 => "0000010100110", 
    457 => "0000010000000", 458 => "0001000111000", 459 => "0000010011000", 
    460 => "0001010000110", 461 => "0000101001110", 462 => "0000010110101", 
    463 => "0000101101100", 464 => "1111011111001", 465 => "0000010110100", 
    466 => "1111101111110", 467 => "1111100011100", 468 => "0001001010000", 
    469 => "1111110111111", 470 => "0000011110110", 471 => "0001001110101", 
    472 => "0001010111000", 473 => "0000010110011", 474 => "0000111000001", 
    475 => "1111011110010", 476 => "0000011011111", 477 => "0000011101101", 
    478 => "1111110000110", 479 => "0001001110111", 480 => "0000000011010", 
    481 => "0000100011010", 482 => "0000011001011", 483 => "0000100000101", 
    484 => "1111111001101", 485 => "1111011011001", 486 => "0000001100000", 
    487 => "0000010000000", 488 => "1111100001100", 489 => "1111101010011", 
    490 => "1111111010001", 491 => "1111111100010", 492 => "1111111111001", 
    493 => "0000001111101", 494 => "0000000111111", 495 => "1111101101100", 
    496 => "1010101101001", 497 => "0000011101001", 498 => "1111001001001", 
    499 => "1111000111100", 500 => "1111101100011", 501 => "1111010011001", 
    502 => "0000001011100", 503 => "1111010110111", 504 => "0000100101001", 
    505 => "1111101001000", 506 => "0000001100000", 507 => "0000001010000", 
    508 => "1111100000100", 509 => "1111111110000", 510 => "1111011101101", 
    511 => "0000011110101" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity mlp_monte_carlo_wg8j is
    generic (
        DataWidth : INTEGER := 13;
        AddressRange : INTEGER := 512;
        AddressWidth : INTEGER := 9);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of mlp_monte_carlo_wg8j is
    component mlp_monte_carlo_wg8j_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    mlp_monte_carlo_wg8j_rom_U :  component mlp_monte_carlo_wg8j_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;

