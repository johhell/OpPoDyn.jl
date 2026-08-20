using PowerDynamics
using OpPoDyn
using OpPoDyn.Library

using PowerDynamics.Library
using ModelingToolkit
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve

using CSV
using DataFrames
using CairoMakie
using Test
using NonlinearSolve: TrustRegion   # init solver needed by test case 2, see there

# ==============================================================================
# BESS_flagtests.jl -- flag-robustness validation against OpenModelica reference
#
# Motivation: BESS.jl only validates WECC_BESS at its DEFAULT flag values
# (L_vplsw=true, PfFlag=Vflag=QFlag=PqFlag=RefFlag=false, VcombFlag=true,
# freqFlag=true). This file validates the model with the flags flipped, using
# as few Modelica reference runs as possible (2 instead of 2^8) while still
# exercising every flag's non-default path at least once -- INCLUDING the
# Vflag&&QFlag compound branch in reec_c, which a naive one-flag-at-a-time
# sweep would never reach (it needs two flags true simultaneously).
#
# Coverage argument (baseline + these two cover every branch):
#   L_vplsw   true (baseline)      / false (both cases)
#   PfFlag    false (baseline)     / true  (both cases)
#   QFlag     false (baseline)     / true  (both cases)
#   Vflag     -- (irrelevant when QFlag=false) / true (case 1) / false (case 2)
#   PqFlag    false (baseline)     / true  (both cases)
#   RefFlag   false (baseline)     / true  (both cases)
#   VcombFlag true (baseline)      / false (both cases)
#   freqFlag  true (baseline)      / true (case 1) / false (case 2)
#
# Two test cases:
#
#   1) "ALLflipped":  L_vplsw=false, PfFlag=true, Vflag=true, QFlag=true,
#                     PqFlag=true, RefFlag=true, VcombFlag=false, freqFlag=true
#      -- exercises the Vflag&&QFlag compound branch (outer Q-PI loop feeds
#         V_con via V_lima).
#
#   2) "ALLflippedExceptVflag": same, but Vflag=false and freqFlag=false
#      -- exercises the !Vflag&&QFlag branch (V_con ~ V_ref0) instead.
#         freqFlag must be false here: with freqFlag=true this combination
#         fails to initialize (InternalLinearSolveFailed).
#         This case additionally needs `subalg=TrustRegion()`: the trigger is
#         PfFlag=true together with Vflag=false and QFlag=true (measured -- with
#         PfFlag=false the default solver converges fine). Default polyalg gives
#         InternalLinearSolveFailed, LevenbergMarquardt gives MaxIters,
#         TrustRegion converges. Same kind of per-combination solver choice that
#         WT4B needs globally (see OpenIPSL_RePSSE_wt).
#
# Reference CSVs exported from OMEdit (OpenIPSL.Tests.Renewable.PSSE.BESSPlant
# with the respective flags on the bESS instance). NOTE on the Modelica side:
# BESSPlant sets `PlantController(fflag = true)` by DEFAULT, so case 1 needs no
# change there while case 2 requires setting fflag=false explicitly.
#
# Careful with the reec_c block names -- they are NOT the same as PV's reec_b,
# because REECCU1 has the SOC limiter as `limiter1`, shifting everything by one:
#            reec_b (PV)            reec_c (BESS)
#   Q_lim    limiter1.y             limiter2.y
#   V_lima   limiter2.y             limiter3.y
#   V_limb   limiter3.y             limiter4.y
#   I_lim    variableLimiter2.y     variableLimiter1.y
# (verified against REECCU1.mo's declarations and connect() statements).
#
# STRUCTURE: the two test cases are flat, self-contained sections (mirroring
# PV.jl / PV_flagtests.jl) so each can be selected and run on its own. Each
# section's @test calls sit in a local @testset wrapped in try/catch, so a
# failing comparison is recorded but does NOT abort before that section's plot
# code runs.
# ==============================================================================

## Tolerances -- see the extended write-up in PV.jl. Short version: after the
## fault clears, the reactive chain (Q_ext -> Iqcmd -> I_q -> pii -> Q_gen)
## settles into a barely-damped oscillation whose amplitude matches the
## reference to a few tenths of a percent while the PHASE drifts between the two
## integrators. RMS then measures phase drift, not model correctness (it varies
## ~50x purely with ODE solver tolerance). Everything off that chain keeps RTOL.
RTOL = 1e-3
RTOL_OSC = 1e-2   # reactive chain only

# ==============================================================================
# Test case 1: "ALLflipped"
# ==============================================================================

@info "Running BESS flag test case: ALLflipped"

ref_bess = CSV.read(
    joinpath(pkgdir(OpPoDyn), "test", "WECC_model_tests", "BESS", "modelica_results_ALLflipped.csv"),
    DataFrame;
    drop=(i, name) -> contains(string(name), "nrows="),
    silencewarnings=true,
)

BESS_BUS = let
    P_0 = 0.015
    Q_0 = -0.056658

    @named BESS = OpPoDyn.Library.WECC_BESS(;
        L_vplsw=false, PfFlag=true, Vflag=true, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=false, freqFlag=true)
    busmodel = MTKBus(BESS; name=:GEN1)
    compile_bus(busmodel, pf=pfPQ(P=P_0, Q=Q_0))
end

sol_bess = OpenIPSL_RePSSE_bess(BESS_BUS; ω_b=2π*50)
ts_bess = refine_timeseries(sol_bess.t)

try
@testset "BESS flagtest ALLflipped" begin

@testset "Kern-Set" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊P_ref), "bESS.PlantController.Pref") < RTOL
    # NOTE (2026-08-17): this passes, but only because RTOL_OSC is loose --
    # measured RMS 0.0085, and the deviation IS visible in the comparison plot.
    # It is a genuine deviation, not phase drift: already 0.0095 before the
    # fault. Q_ext starts at -0.04655 instead of -0.05666 and then ramps at
    # +0.0054/s instead of the reference's +0.0255/s.
    # Trigger isolated by flipping single flags (Julia-side only, no new
    # reference runs needed): it is Vflag=true. With Vflag=false and everything
    # else unchanged, Q_ext lands on -0.05666 and ramps at +0.0259/s, i.e. it
    # matches -- which is also why test case 2 below (Vflag=false) is clean at
    # RMS 0.00055. freqFlag was ruled OUT: toggling it leaves Q_ext bit-identical.
    # Harmless for the model itself: PfFlag=true means reec_c takes Q_con from
    # the power-factor branch and never reads Qext_in, so repca's Q chain is a
    # dead branch here and every downstream quantity still matches.
    # OPEN: why Vflag=true (which lives in reec_c's outer Q-PI, upstream of the
    # network feedback into repca) shifts repca's Q_ext this way.
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊Q_ext), "bESS.PlantController.Qext") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊Q_gen), "bESS.RenewableController.Qgen") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊P_gen), "bESS.RenewableController.Pe") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊V_t), "bESS.RenewableController.Vt") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pcmd), "bESS.RenewableController.Ipcmd") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qcmd), "bESS.RenewableController.Iqcmd") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊soc_lim), "bESS.RenewableController.sOC_logic.SOC") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pmax), "bESS.RenewableController.IPMAX.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pmin), "bESS.RenewableController.IPMIN.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qmax), "bESS.RenewableController.IQMAX.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qmin), "bESS.RenewableController.IQMIN.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_lvpl), "bESS.RenewableGenerator.LVPL.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pii), "bESS.RenewableGenerator.p.ii") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pir), "bESS.RenewableGenerator.p.ir") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pvi), "bESS.RenewableGenerator.p.vi") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pvr), "bESS.RenewableGenerator.p.vr") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_p), "bESS.RenewableGenerator.IP.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_q), "bESS.RenewableGenerator.IOLIM.y") < RTOL_OSC
end

@testset "PfFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊P_PF), "bESS.RenewableController.simpleLag1.y") < RTOL
end

@testset "Vflag&&QFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊Q_lim), "bESS.RenewableController.limiter2.y") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊V_lima), "bESS.RenewableController.limiter3.y") < RTOL
end

@testset "QFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊V_limb), "bESS.RenewableController.limiter4.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_lim), "bESS.RenewableController.variableLimiter1.y") < RTOL_OSC
end

@testset "RefFlag/VcombFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊ΔQ_in), "bESS.PlantController.REFFLAG.y") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊V_in), "bESS.PlantController.VCFLAG.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊V_droop), "bESS.PlantController.add3.y") < RTOL
end

@testset "freqFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊P_e), "bESS.PlantController.add2.y") < RTOL
end

## Plot layout follows the actual signal flow: repca (plant controller) first,
## in the order its own equations compute things, then reecc (electrical
## controller), then regca (generator) and the terminal quantities.
if isdefined(Main, :EXPORT_FIGURES) && Main.EXPORT_FIGURES
    fig_ALLflipped = let
        fig = Figure(size=(1400, 2600))

        # --- repca (Plant Controller) ---
        ax1 = Axis(fig[1,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: V_droop & V_in (VcombFlag)")
        lines!(ax1, ref_bess.time, ref_bess[!, "bESS.PlantController.add3.y"]; label="OpenIPSL V_droop", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax1, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊V_droop)).u; label="PD V_droop", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax1, ref_bess.time, ref_bess[!, "bESS.PlantController.VCFLAG.y"]; label="OpenIPSL V_in", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax1, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊V_in)).u; label="PD V_in", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax1; position=:rb)

        ax2 = Axis(fig[1,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: ΔQ_in (RefFlag)")
        lines!(ax2, ref_bess.time, ref_bess[!, "bESS.PlantController.REFFLAG.y"]; label="OpenIPSL ΔQ_in", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax2, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊ΔQ_in)).u; label="PD ΔQ_in", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax2; position=:rb)

        ax3 = Axis(fig[2,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: Q_ext & P_ref (final outputs)")
        lines!(ax3, ref_bess.time, ref_bess[!, "bESS.PlantController.Qext"]; label="OpenIPSL Qext", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax3, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊Q_ext)).u; label="PD Qext", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax3, ref_bess.time, ref_bess[!, "bESS.PlantController.Pref"]; label="OpenIPSL Pref", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax3, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊P_ref)).u; label="PD Pref", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax3; position=:rb)

        ax4 = Axis(fig[2,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: P_e (freqFlag)")
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.PlantController.add2.y"]; label="OpenIPSL P_e", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊P_e)).u; label="PD P_e", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax4; position=:rb)

        # --- reecc (Electrical Controller) ---
        ax5 = Axis(fig[3,1]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Q_gen, P_gen & Vt (measured inputs)")
        lines!(ax5, ref_bess.time, ref_bess[!, "bESS.RenewableController.Qgen"]; label="OpenIPSL Q_gen", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax5, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊Q_gen)).u; label="PD Q_gen", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax5, ref_bess.time, ref_bess[!, "bESS.RenewableController.Pe"]; label="OpenIPSL P_gen", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax5, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊P_gen)).u; label="PD P_gen", color=Cycled(2), linestyle=:dash, linewidth=2)
        lines!(ax5, ref_bess.time, ref_bess[!, "bESS.RenewableController.Vt"]; label="OpenIPSL Vt", color=Cycled(3), linewidth=2, alpha=0.6)
        lines!(ax5, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊V_t)).u; label="PD Vt", color=Cycled(3), linestyle=:dash, linewidth=2)
        axislegend(ax5; position=:rb)

        ax6 = Axis(fig[3,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: P_PF (PfFlag)")
        lines!(ax6, ref_bess.time, ref_bess[!, "bESS.RenewableController.simpleLag1.y"]; label="OpenIPSL P_PF", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax6, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊P_PF)).u; label="PD P_PF", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax6; position=:rb)

        ax7 = Axis(fig[4,1]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Q_lim & V_lima (Vflag&&QFlag)")
        lines!(ax7, ref_bess.time, ref_bess[!, "bESS.RenewableController.limiter2.y"]; label="OpenIPSL Q_lim", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax7, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊Q_lim)).u; label="PD Q_lim", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax7, ref_bess.time, ref_bess[!, "bESS.RenewableController.limiter3.y"]; label="OpenIPSL V_lima", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax7, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊V_lima)).u; label="PD V_lima", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax7; position=:rb)

        ax8 = Axis(fig[4,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: V_limb & I_lim (QFlag)")
        lines!(ax8, ref_bess.time, ref_bess[!, "bESS.RenewableController.limiter4.y"]; label="OpenIPSL V_limb", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax8, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊V_limb)).u; label="PD V_limb", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax8, ref_bess.time, ref_bess[!, "bESS.RenewableController.variableLimiter1.y"]; label="OpenIPSL I_lim", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax8, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_lim)).u; label="PD I_lim", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax8; position=:rb)

        ax9 = Axis(fig[5,1]; xlabel="Time [s]", ylabel="SOC [-]", title="reecc: soc_lim (BESS storage state)")
        lines!(ax9, ref_bess.time, ref_bess[!, "bESS.RenewableController.sOC_logic.SOC"]; label="OpenIPSL SOC", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax9, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊soc_lim)).u; label="PD SOC", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax9; position=:rb)

        ax10 = Axis(fig[5,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Ipcmd & Iqcmd (final outputs)")
        lines!(ax10, ref_bess.time, ref_bess[!, "bESS.RenewableController.Ipcmd"]; label="OpenIPSL Ipcmd", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax10, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_pcmd)).u; label="PD Ipcmd", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax10, ref_bess.time, ref_bess[!, "bESS.RenewableController.Iqcmd"]; label="OpenIPSL Iqcmd", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax10, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_qcmd)).u; label="PD Iqcmd", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax10; position=:rb)

        # --- regca (Generator) + terminal ---
        ax11 = Axis(fig[6,1]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_lvpl")
        lines!(ax11, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.LVPL.y"]; label="OpenIPSL I_lvpl", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax11, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_lvpl)).u; label="PD I_lvpl", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax11; position=:rb)

        ax12 = Axis(fig[6,2]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_p & I_q (final outputs)")
        lines!(ax12, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.IP.y"]; label="OpenIPSL I_p", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax12, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_p)).u; label="PD I_p", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax12, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.IOLIM.y"]; label="OpenIPSL I_q", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax12, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_q)).u; label="PD I_q", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax12; position=:rb)

        ax13 = Axis(fig[7,1]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pir & pii")
        lines!(ax13, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.ir"]; label="OpenIPSL pir", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax13, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pir)).u; label="PD pir", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax13, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.ii"]; label="OpenIPSL pii", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax13, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pii)).u; label="PD pii", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax13; position=:rb)

        ax14 = Axis(fig[7,2]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pvr & pvi")
        lines!(ax14, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.vr"]; label="OpenIPSL pvr", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax14, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pvr)).u; label="PD pvr", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax14, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.vi"]; label="OpenIPSL pvi", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax14, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pvi)).u; label="PD pvi", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax14; position=:rb)

        Label(fig[0, :], "BESS flag test case: ALLflipped  --  L_vplsw=false, PfFlag=true, Vflag=true, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=false, freqFlag=true"; fontsize=14)

        fig
    end
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "BESS_flagtest_ALLflipped_comparison.png"), fig_ALLflipped)
end

end # @testset "BESS flagtest ALLflipped"
catch e
    @warn "BESS flagtest ALLflipped: exception escaped the section (should only happen on a real error, not a plain @test failure)" exception=(e, catch_backtrace())
end


# ==============================================================================
# Test case 2: "ALLflippedExceptVflag"
# ==============================================================================

@info "Running BESS flag test case: ALLflippedExceptVflag"

ref_bess = CSV.read(
    joinpath(pkgdir(OpPoDyn), "test", "WECC_model_tests", "BESS", "modelica_results_ALLflippedExceptVflag.csv"),
    DataFrame;
    drop=(i, name) -> contains(string(name), "nrows="),
    silencewarnings=true,
)

BESS_BUS = let
    P_0 = 0.015
    Q_0 = -0.056658

    @named BESS = OpPoDyn.Library.WECC_BESS(;
        L_vplsw=false, PfFlag=true, Vflag=false, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=false, freqFlag=false)
    busmodel = MTKBus(BESS; name=:GEN1)
    compile_bus(busmodel, pf=pfPQ(P=P_0, Q=Q_0))
end

sol_bess = OpenIPSL_RePSSE_bess(BESS_BUS; ω_b=2π*50, subalg=TrustRegion())
ts_bess = refine_timeseries(sol_bess.t)

try
@testset "BESS flagtest ALLflippedExceptVflag" begin

@testset "Kern-Set" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊P_ref), "bESS.PlantController.Pref") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊Q_ext), "bESS.PlantController.Qext") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊Q_gen), "bESS.RenewableController.Qgen") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊P_gen), "bESS.RenewableController.Pe") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊V_t), "bESS.RenewableController.Vt") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pcmd), "bESS.RenewableController.Ipcmd") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qcmd), "bESS.RenewableController.Iqcmd") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊soc_lim), "bESS.RenewableController.sOC_logic.SOC") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pmax), "bESS.RenewableController.IPMAX.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pmin), "bESS.RenewableController.IPMIN.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qmax), "bESS.RenewableController.IQMAX.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qmin), "bESS.RenewableController.IQMIN.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_lvpl), "bESS.RenewableGenerator.LVPL.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pii), "bESS.RenewableGenerator.p.ii") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pir), "bESS.RenewableGenerator.p.ir") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pvi), "bESS.RenewableGenerator.p.vi") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pvr), "bESS.RenewableGenerator.p.vr") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_p), "bESS.RenewableGenerator.IP.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_q), "bESS.RenewableGenerator.IOLIM.y") < RTOL_OSC
end

@testset "PfFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊P_PF), "bESS.RenewableController.simpleLag1.y") < RTOL
end

## no Vflag&&QFlag testset here: Vflag=false -> Q_lim / V_lima don't exist in the
## Julia model (the reference CSV still contains limiter2.y / limiter3.y, since
## Modelica always instantiates those blocks -- they are simply not switched
## through -- but there is nothing on the Julia side to compare them against).

@testset "QFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊V_limb), "bESS.RenewableController.limiter4.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_lim), "bESS.RenewableController.variableLimiter1.y") < RTOL_OSC
end

@testset "RefFlag/VcombFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊ΔQ_in), "bESS.PlantController.REFFLAG.y") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊V_in), "bESS.PlantController.VCFLAG.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊V_droop), "bESS.PlantController.add3.y") < RTOL
end

## no freqFlag testset here: freqFlag=false -> P_e doesn't exist in the Julia model

if isdefined(Main, :EXPORT_FIGURES) && Main.EXPORT_FIGURES
    fig_ALLflippedExceptVflag = let
        fig = Figure(size=(1400, 2400))

        # --- repca (Plant Controller) ---
        ax1 = Axis(fig[1,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: V_droop & V_in (VcombFlag)")
        lines!(ax1, ref_bess.time, ref_bess[!, "bESS.PlantController.add3.y"]; label="OpenIPSL V_droop", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax1, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊V_droop)).u; label="PD V_droop", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax1, ref_bess.time, ref_bess[!, "bESS.PlantController.VCFLAG.y"]; label="OpenIPSL V_in", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax1, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊V_in)).u; label="PD V_in", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax1; position=:rb)

        ax2 = Axis(fig[1,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: ΔQ_in (RefFlag)")
        lines!(ax2, ref_bess.time, ref_bess[!, "bESS.PlantController.REFFLAG.y"]; label="OpenIPSL ΔQ_in", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax2, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊ΔQ_in)).u; label="PD ΔQ_in", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax2; position=:rb)

        ax3 = Axis(fig[2,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: Q_ext & P_ref (final outputs)")
        lines!(ax3, ref_bess.time, ref_bess[!, "bESS.PlantController.Qext"]; label="OpenIPSL Qext", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax3, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊Q_ext)).u; label="PD Qext", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax3, ref_bess.time, ref_bess[!, "bESS.PlantController.Pref"]; label="OpenIPSL Pref", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax3, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊P_ref)).u; label="PD Pref", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax3; position=:rb)

        # --- reecc (Electrical Controller); no P_e / Q_lim / V_lima panels here ---
        ax4 = Axis(fig[2,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Q_gen, P_gen & Vt (measured inputs)")
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.RenewableController.Qgen"]; label="OpenIPSL Q_gen", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊Q_gen)).u; label="PD Q_gen", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.RenewableController.Pe"]; label="OpenIPSL P_gen", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊P_gen)).u; label="PD P_gen", color=Cycled(2), linestyle=:dash, linewidth=2)
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.RenewableController.Vt"]; label="OpenIPSL Vt", color=Cycled(3), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊V_t)).u; label="PD Vt", color=Cycled(3), linestyle=:dash, linewidth=2)
        axislegend(ax4; position=:rb)

        ax5 = Axis(fig[3,1]; xlabel="Time [s]", ylabel="[pu]", title="reecc: P_PF (PfFlag)")
        lines!(ax5, ref_bess.time, ref_bess[!, "bESS.RenewableController.simpleLag1.y"]; label="OpenIPSL P_PF", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax5, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊P_PF)).u; label="PD P_PF", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax5; position=:rb)

        ax6 = Axis(fig[3,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: V_limb & I_lim (QFlag)")
        lines!(ax6, ref_bess.time, ref_bess[!, "bESS.RenewableController.limiter4.y"]; label="OpenIPSL V_limb", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax6, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊V_limb)).u; label="PD V_limb", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax6, ref_bess.time, ref_bess[!, "bESS.RenewableController.variableLimiter1.y"]; label="OpenIPSL I_lim", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax6, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_lim)).u; label="PD I_lim", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax6; position=:rb)

        ax7 = Axis(fig[4,1]; xlabel="Time [s]", ylabel="SOC [-]", title="reecc: soc_lim (BESS storage state)")
        lines!(ax7, ref_bess.time, ref_bess[!, "bESS.RenewableController.sOC_logic.SOC"]; label="OpenIPSL SOC", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax7, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊soc_lim)).u; label="PD SOC", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax7; position=:rb)

        ax8 = Axis(fig[4,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Ipcmd & Iqcmd (final outputs)")
        lines!(ax8, ref_bess.time, ref_bess[!, "bESS.RenewableController.Ipcmd"]; label="OpenIPSL Ipcmd", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax8, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_pcmd)).u; label="PD Ipcmd", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax8, ref_bess.time, ref_bess[!, "bESS.RenewableController.Iqcmd"]; label="OpenIPSL Iqcmd", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax8, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_qcmd)).u; label="PD Iqcmd", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax8; position=:rb)

        # --- regca (Generator) + terminal ---
        ax9 = Axis(fig[5,1]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_lvpl")
        lines!(ax9, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.LVPL.y"]; label="OpenIPSL I_lvpl", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax9, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_lvpl)).u; label="PD I_lvpl", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax9; position=:rb)

        ax10 = Axis(fig[5,2]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_p & I_q (final outputs)")
        lines!(ax10, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.IP.y"]; label="OpenIPSL I_p", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax10, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_p)).u; label="PD I_p", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax10, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.IOLIM.y"]; label="OpenIPSL I_q", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax10, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_q)).u; label="PD I_q", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax10; position=:rb)

        ax11 = Axis(fig[6,1]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pir & pii")
        lines!(ax11, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.ir"]; label="OpenIPSL pir", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax11, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pir)).u; label="PD pir", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax11, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.ii"]; label="OpenIPSL pii", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax11, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pii)).u; label="PD pii", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax11; position=:rb)

        ax12 = Axis(fig[6,2]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pvr & pvi")
        lines!(ax12, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.vr"]; label="OpenIPSL pvr", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax12, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pvr)).u; label="PD pvr", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax12, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.vi"]; label="OpenIPSL pvi", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax12, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pvi)).u; label="PD pvi", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax12; position=:rb)

        Label(fig[0, :], "BESS flag test case: ALLflippedExceptVflag  --  L_vplsw=false, PfFlag=true, Vflag=false, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=false, freqFlag=false"; fontsize=14)

        fig
    end
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "BESS_flagtest_ALLflippedExceptVflag_comparison.png"), fig_ALLflippedExceptVflag)
end

end # @testset "BESS flagtest ALLflippedExceptVflag"
catch e
    @warn "BESS flagtest ALLflippedExceptVflag: exception escaped the section (should only happen on a real error, not a plain @test failure)" exception=(e, catch_backtrace())
end


# ==============================================================================
# Test case 3: "QFlagFalseRefFlag"
# ==============================================================================
#
# Motivation: Q_ext (repca's reactive output) only actually DRIVES anything when
# PfFlag=false AND QFlag=false:
#     PfFlag=true  -> Q_con ~ P_PF*tan(P_faref.u), Qext_in is never read
#     QFlag=true   -> I_qcon ~ I_lim, so I_qin (fed by Q_con) is unused
# Test cases 1 and 2 above both have QFlag=true (and PfFlag=true), so Q_ext is a
# structurally dead branch there -- which is why its visible deviation in case 1
# (see the note on that test) leaves every downstream quantity intact, but also
# why that comparison cannot be judged. The baseline (BESS.jl) does exercise
# Q_ext for real (PfFlag=QFlag=false), but only with RefFlag=false (ΔQ_in ~ ΔQ).
#
# This case closes the gap: PfFlag=false AND QFlag=false (Q_ext reaches the
# terminal quantities) TOGETHER WITH RefFlag=true (ΔQ_in ~ ΔV). The analogous
# PV case is what uncovered the T_fltr bug in repc_a; the analogous WT4B case
# passes 24/24 and showed the RefFlag=true path to be correct there.
#
# Verified before the reference was generated: this combination initializes and
# simulates with the default init solver (no subalg needed).
#
# Flags: L_vplsw=false, PfFlag=false, Vflag=false, QFlag=false, PqFlag=true,
#        RefFlag=true, VcombFlag=false, freqFlag=false
#
# With QFlag=false, PfFlag=false and freqFlag=false, the quantities P_PF /
# Q_lim / V_lima / V_limb / I_lim / P_e do not exist in the Julia model, so
# there are no PfFlag-, Vflag&&QFlag-, QFlag- or freqFlag-specific testsets
# here (the reference CSV may still contain those columns, since Modelica
# always instantiates the blocks).
# ==============================================================================

@info "Running BESS flag test case: QFlagFalseRefFlag"

ref_bess = CSV.read(
    joinpath(pkgdir(OpPoDyn), "test", "WECC_model_tests", "BESS", "modelica_results_QFlagFalseRefFlag.csv"),
    DataFrame;
    drop=(i, name) -> contains(string(name), "nrows="),
    silencewarnings=true,
)

BESS_BUS = let
    @named BESS = OpPoDyn.Library.WECC_BESS(;
        L_vplsw=false, PfFlag=false, Vflag=false, QFlag=false, PqFlag=true,
        RefFlag=true, VcombFlag=false, freqFlag=false)
    busmodel = MTKBus(BESS; name=:GEN1)
    compile_bus(busmodel, pf=pfPQ(P=0.015, Q=-0.056658))
end

sol_bess = OpenIPSL_RePSSE_bess(BESS_BUS; ω_b=2π*50)   # default init solver
ts_bess = refine_timeseries(sol_bess.t)

try
@testset "BESS flagtest QFlagFalseRefFlag" begin

@testset "Kern-Set" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊P_ref), "bESS.PlantController.Pref") < RTOL
    # Unlike in test cases 1 and 2, Q_ext is NOT a dead branch here -- it feeds
    # Q_con -> I_t -> I_qin -> I_qcon and thus the terminal quantities. So this
    # comparison is meaningful, and a failure here would be a real finding.
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊Q_ext), "bESS.PlantController.Qext") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊Q_gen), "bESS.RenewableController.Qgen") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊P_gen), "bESS.RenewableController.Pe") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊V_t), "bESS.RenewableController.Vt") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pcmd), "bESS.RenewableController.Ipcmd") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qcmd), "bESS.RenewableController.Iqcmd") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊soc_lim), "bESS.RenewableController.sOC_logic.SOC") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pmax), "bESS.RenewableController.IPMAX.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_pmin), "bESS.RenewableController.IPMIN.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qmax), "bESS.RenewableController.IQMAX.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊reecc₊I_qmin), "bESS.RenewableController.IQMIN.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_lvpl), "bESS.RenewableGenerator.LVPL.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pii), "bESS.RenewableGenerator.p.ii") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pir), "bESS.RenewableGenerator.p.ir") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pvi), "bESS.RenewableGenerator.p.vi") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊pvr), "bESS.RenewableGenerator.p.vr") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_p), "bESS.RenewableGenerator.IP.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊regca₊I_q), "bESS.RenewableGenerator.IOLIM.y") < RTOL_OSC
end

@testset "RefFlag/VcombFlag-specific" begin
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊ΔQ_in), "bESS.PlantController.REFFLAG.y") < RTOL_OSC
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊V_in), "bESS.PlantController.VCFLAG.y") < RTOL
    @test ref_rms_error(sol_bess, ref_bess, VIndex(:GEN1, :BESS₊repca₊V_droop), "bESS.PlantController.add3.y") < RTOL
end

if isdefined(Main, :EXPORT_FIGURES) && Main.EXPORT_FIGURES
    fig_QFlagFalseRefFlag = let
        fig = Figure(size=(1400, 2000))

        ax1 = Axis(fig[1,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: V_droop & V_in (VcombFlag)")
        lines!(ax1, ref_bess.time, ref_bess[!, "bESS.PlantController.add3.y"]; label="OpenIPSL V_droop", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax1, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊V_droop)).u; label="PD V_droop", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax1, ref_bess.time, ref_bess[!, "bESS.PlantController.VCFLAG.y"]; label="OpenIPSL V_in", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax1, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊V_in)).u; label="PD V_in", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax1; position=:rb)

        ax2 = Axis(fig[1,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: ΔQ_in (RefFlag)")
        lines!(ax2, ref_bess.time, ref_bess[!, "bESS.PlantController.REFFLAG.y"]; label="OpenIPSL ΔQ_in", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax2, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊ΔQ_in)).u; label="PD ΔQ_in", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax2; position=:rb)

        ax3 = Axis(fig[2,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: Q_ext & P_ref (ACTIVE here, not a dead branch)")
        lines!(ax3, ref_bess.time, ref_bess[!, "bESS.PlantController.Qext"]; label="OpenIPSL Qext", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax3, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊Q_ext)).u; label="PD Qext", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax3, ref_bess.time, ref_bess[!, "bESS.PlantController.Pref"]; label="OpenIPSL Pref", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax3, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊repca₊P_ref)).u; label="PD Pref", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax3; position=:rb)

        ax4 = Axis(fig[2,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Q_gen, P_gen & Vt (measured inputs)")
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.RenewableController.Qgen"]; label="OpenIPSL Q_gen", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊Q_gen)).u; label="PD Q_gen", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.RenewableController.Pe"]; label="OpenIPSL P_gen", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊P_gen)).u; label="PD P_gen", color=Cycled(2), linestyle=:dash, linewidth=2)
        lines!(ax4, ref_bess.time, ref_bess[!, "bESS.RenewableController.Vt"]; label="OpenIPSL Vt", color=Cycled(3), linewidth=2, alpha=0.6)
        lines!(ax4, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊V_t)).u; label="PD Vt", color=Cycled(3), linestyle=:dash, linewidth=2)
        axislegend(ax4; position=:rb)

        ax5 = Axis(fig[3,1]; xlabel="Time [s]", ylabel="SOC [-]", title="reecc: soc_lim (BESS storage state)")
        lines!(ax5, ref_bess.time, ref_bess[!, "bESS.RenewableController.sOC_logic.SOC"]; label="OpenIPSL SOC", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax5, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊soc_lim)).u; label="PD SOC", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax5; position=:rb)

        ax6 = Axis(fig[3,2]; xlabel="Time [s]", ylabel="[pu]", title="reecc: Ipcmd & Iqcmd (final outputs)")
        lines!(ax6, ref_bess.time, ref_bess[!, "bESS.RenewableController.Ipcmd"]; label="OpenIPSL Ipcmd", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax6, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_pcmd)).u; label="PD Ipcmd", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax6, ref_bess.time, ref_bess[!, "bESS.RenewableController.Iqcmd"]; label="OpenIPSL Iqcmd", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax6, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊reecc₊I_qcmd)).u; label="PD Iqcmd", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax6; position=:rb)

        ax7 = Axis(fig[4,1]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_lvpl")
        lines!(ax7, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.LVPL.y"]; label="OpenIPSL I_lvpl", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax7, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_lvpl)).u; label="PD I_lvpl", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax7; position=:rb)

        ax8 = Axis(fig[4,2]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_p & I_q (final outputs)")
        lines!(ax8, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.IP.y"]; label="OpenIPSL I_p", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax8, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_p)).u; label="PD I_p", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax8, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.IOLIM.y"]; label="OpenIPSL I_q", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax8, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊regca₊I_q)).u; label="PD I_q", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax8; position=:rb)

        ax9 = Axis(fig[5,1]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pir & pii")
        lines!(ax9, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.ir"]; label="OpenIPSL pir", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax9, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pir)).u; label="PD pir", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax9, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.ii"]; label="OpenIPSL pii", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax9, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pii)).u; label="PD pii", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax9; position=:rb)

        ax10 = Axis(fig[5,2]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pvr & pvi")
        lines!(ax10, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.vr"]; label="OpenIPSL pvr", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax10, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pvr)).u; label="PD pvr", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax10, ref_bess.time, ref_bess[!, "bESS.RenewableGenerator.p.vi"]; label="OpenIPSL pvi", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax10, ts_bess, sol_bess(ts_bess, idxs=VIndex(:GEN1, :BESS₊pvi)).u; label="PD pvi", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax10; position=:rb)

        Label(fig[0, :], "BESS flag test case: QFlagFalseRefFlag  --  L_vplsw=false, PfFlag=false, Vflag=false, QFlag=false, PqFlag=true, RefFlag=true, VcombFlag=false, freqFlag=false"; fontsize=13)

        fig
    end
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "BESS_flagtest_QFlagFalseRefFlag_comparison.png"), fig_QFlagFalseRefFlag)
end

end # @testset "BESS flagtest QFlagFalseRefFlag"
catch e
    @warn "BESS flagtest QFlagFalseRefFlag: exception escaped the section (should only happen on a real error, not a plain @test failure)" exception=(e, catch_backtrace())
end
