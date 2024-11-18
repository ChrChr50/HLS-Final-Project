// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and OpenCL
// Version: 2019.2
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

#ifndef _dut_HH_
#define _dut_HH_

#include "systemc.h"
#include "AESL_pkg.h"

#include "mlp_monte_carlo.h"
#include "dut_input_V.h"

namespace ap_rtl {

struct dut : public sc_module {
    // Port declarations 12
    sc_in_clk ap_clk;
    sc_in< sc_logic > ap_rst;
    sc_in< sc_logic > ap_start;
    sc_out< sc_logic > ap_done;
    sc_out< sc_logic > ap_idle;
    sc_out< sc_logic > ap_ready;
    sc_in< sc_lv<16> > strm_in_V_V_dout;
    sc_in< sc_logic > strm_in_V_V_empty_n;
    sc_out< sc_logic > strm_in_V_V_read;
    sc_out< sc_lv<16> > strm_out_V_V_din;
    sc_in< sc_logic > strm_out_V_V_full_n;
    sc_out< sc_logic > strm_out_V_V_write;


    // Module declarations
    dut(sc_module_name name);
    SC_HAS_PROCESS(dut);

    ~dut();

    sc_trace_file* mVcdFile;

    ofstream mHdltvinHandle;
    ofstream mHdltvoutHandle;
    dut_input_V* input_V_U;
    mlp_monte_carlo* grp_mlp_monte_carlo_fu_103;
    sc_signal< sc_lv<4> > ap_CS_fsm;
    sc_signal< sc_logic > ap_CS_fsm_state1;
    sc_signal< sc_logic > strm_in_V_V_blk_n;
    sc_signal< sc_logic > ap_CS_fsm_state2;
    sc_signal< sc_lv<1> > icmp_ln22_fu_128_p2;
    sc_signal< sc_logic > strm_out_V_V_blk_n;
    sc_signal< sc_logic > ap_CS_fsm_state3;
    sc_signal< sc_logic > ap_CS_fsm_state4;
    sc_signal< sc_lv<4> > i_fu_134_p2;
    sc_signal< bool > ap_block_state2;
    sc_signal< sc_lv<16> > variance_output_V_reg_162;
    sc_signal< sc_logic > grp_mlp_monte_carlo_fu_103_ap_ready;
    sc_signal< sc_logic > grp_mlp_monte_carlo_fu_103_ap_done;
    sc_signal< sc_lv<4> > input_V_address0;
    sc_signal< sc_logic > input_V_ce0;
    sc_signal< sc_logic > input_V_we0;
    sc_signal< sc_lv<16> > input_V_q0;
    sc_signal< sc_logic > grp_mlp_monte_carlo_fu_103_ap_start;
    sc_signal< sc_logic > grp_mlp_monte_carlo_fu_103_ap_idle;
    sc_signal< sc_lv<4> > grp_mlp_monte_carlo_fu_103_input_V_address0;
    sc_signal< sc_logic > grp_mlp_monte_carlo_fu_103_input_V_ce0;
    sc_signal< sc_lv<16> > grp_mlp_monte_carlo_fu_103_ap_return_0;
    sc_signal< sc_lv<16> > grp_mlp_monte_carlo_fu_103_ap_return_1;
    sc_signal< sc_lv<4> > i_0_reg_92;
    sc_signal< sc_logic > grp_mlp_monte_carlo_fu_103_ap_start_reg;
    sc_signal< bool > ap_block_state2_ignore_call0;
    sc_signal< sc_lv<64> > zext_ln23_fu_140_p1;
    sc_signal< sc_lv<4> > ap_NS_fsm;
    static const sc_logic ap_const_logic_1;
    static const sc_logic ap_const_logic_0;
    static const sc_lv<4> ap_ST_fsm_state1;
    static const sc_lv<4> ap_ST_fsm_state2;
    static const sc_lv<4> ap_ST_fsm_state3;
    static const sc_lv<4> ap_ST_fsm_state4;
    static const sc_lv<32> ap_const_lv32_0;
    static const sc_lv<32> ap_const_lv32_1;
    static const sc_lv<1> ap_const_lv1_0;
    static const sc_lv<32> ap_const_lv32_2;
    static const sc_lv<32> ap_const_lv32_3;
    static const sc_lv<4> ap_const_lv4_0;
    static const sc_lv<1> ap_const_lv1_1;
    static const sc_lv<4> ap_const_lv4_9;
    static const sc_lv<4> ap_const_lv4_1;
    static const bool ap_const_boolean_1;
    // Thread declarations
    void thread_ap_clk_no_reset_();
    void thread_ap_CS_fsm_state1();
    void thread_ap_CS_fsm_state2();
    void thread_ap_CS_fsm_state3();
    void thread_ap_CS_fsm_state4();
    void thread_ap_block_state2();
    void thread_ap_block_state2_ignore_call0();
    void thread_ap_done();
    void thread_ap_idle();
    void thread_ap_ready();
    void thread_grp_mlp_monte_carlo_fu_103_ap_start();
    void thread_i_fu_134_p2();
    void thread_icmp_ln22_fu_128_p2();
    void thread_input_V_address0();
    void thread_input_V_ce0();
    void thread_input_V_we0();
    void thread_strm_in_V_V_blk_n();
    void thread_strm_in_V_V_read();
    void thread_strm_out_V_V_blk_n();
    void thread_strm_out_V_V_din();
    void thread_strm_out_V_V_write();
    void thread_zext_ln23_fu_140_p1();
    void thread_ap_NS_fsm();
    void thread_hdltv_gen();
};

}

using namespace ap_rtl;

#endif
