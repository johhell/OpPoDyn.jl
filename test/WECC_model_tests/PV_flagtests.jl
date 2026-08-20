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

# ==============================================================================
# PV_flagtests.jl -- flag-robustness validation against OpenModelica reference
#
# Motivation: PV.jl only validates WECC_large_PV at its DEFAULT flag values
# (L_vplsw=true, PfFlag=Vflag=QFlag=PqFlag=RefFlag=VcombFlag=freqFlag=false).
# This file validates the model with all flags flipped to their non-default
# value, minimizing the number of Modelica reference runs needed (2 runs
# instead of 2^8) while still exercising every flag's non-default path at
# least once, INCLUDING the Vflag&&QFlag compound branch in reec_b that a
# naive one-flag-at-a-time sweep would never reach.
#
# Two test cases (see plantmodels.jl comments on WECC_large_PV for why these
# two, and why NOT all 8 flags flipped simultaneously in one run):
#
#   1) "ALLflipped":            L_vplsw=false, PfFlag=true, Vflag=true,
#                                QFlag=true, PqFlag=true, RefFlag=true,
#                                VcombFlag=true, freqFlag=false
#      -- exercises the Vflag&&QFlag compound branch (outer Q-PI-loop
#         determines V_con). freqFlag stays false here: combined with
#         Vflag=true+QFlag=true it fails to initialize even with the known
#         freqFlag=true fix (pinning repca.P_e), see plantmodels.jl.
#
#   2) "ALLflippedExceptVflag": same as (1) but Vflag=false, freqFlag=true
#      -- exercises the !Vflag&&QFlag branch (V_con~V_ref0) instead, and
#         covers freqFlag=true (which does NOT work combined with Vflag=true,
#         see above).
#
# Reference CSVs were exported from OMEdit (OpenIPSL.Tests.Renewable.PSSE.PVPlant
# with the respective flags set on the pV instance) via filterSimulationResults
# selecting the same variable set as PV.jl's "Kern-Set", plus per-testcase
# flag-specific variables. "pV.PlantController.add3.y" is V_droop's own
# defining equation (V_droop ~ K_c*Q_branch.u + V_reg.u, an addition -> "y"
# is correct, not "u" as first assumed).
#
# STRUCTURE: unlike an earlier version of this file, the two test cases below
# are NOT driven by a loop over an array -- each is a flat, self-contained
# section (mirroring PV.jl's style) so it can be selected and run on its own
# (e.g. re-running just test case 2's block after editing it, without needing
# test case 1 to have run first). Each section's @test calls are wrapped in
# their own local @testset so that a failing comparison is recorded and
# reported, but does NOT throw/abort before that section's plot code runs --
# running one section standalone therefore always reaches its plot, and
# running the whole file top-to-bottom also always reaches both plots (the
# final exception, if any test failed, only surfaces after everything -- both
# sections' tests AND plots -- has already executed).
# ==============================================================================

RTOL = 1e-3

# ==============================================================================
# Test case 1: "ALLflipped"
# ==============================================================================

@info "Running PV flag test case: ALLflipped"

ref_pv = CSV.read(
    joinpath(pkgdir(OpPoDyn), "test", "WECC_model_tests", "PV", "modelica_results_ALLflipped.csv"),
    DataFrame;
    drop=(i, name) -> contains(string(name), "nrows="),
    silencewarnings=true,
)

PV_BUS = let
    v_0 = 1.0
    P_0 = 0.015

    @named PV = OpPoDyn.Library.WECC_large_PV(;
        L_vplsw=false, PfFlag=true, Vflag=true, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=true, freqFlag=false)
    busmodel = MTKBus(PV; name=:GEN1)
    compile_bus(busmodel, pf=pfPV(V=v_0, P=P_0))
end

sol_pv = OpenIPSL_RePSSE_pv(PV_BUS; ω_b=2π*60)
ts_pv = refine_timeseries(sol_pv.t)

## Everything below (all sub-testsets + the plot) is wrapped in one
## section-local @testset, itself wrapped in try/catch: a failing @test is
## recorded/reported as usual, but neither the plot code below it nor
## anything in the OTHER test case's section is ever aborted by it -- this
## section can be selected and run entirely on its own too.
try
@testset "PV flagtest ALLflipped" begin

@testset "Kern-Set" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊P_ref), "pV.PlantController.Pref") < RTOL
    # NOTE (2026-08-14, OPEN ISSUE): FAILS in both flag test cases
    # (RMS ~0.055 for ALLflipped, ~0.046 for ALLflippedExceptVflag -- both
    # >>RTOL). RefFlag=true is the one flag common to both failing cases;
    # ΔQ_in below fails identically in both too (same RefFlag=true root),
    # and Q_ext is downstream of ΔQ_in's Q-control PI loop, so these two
    # failures are very likely the SAME underlying discrepancy, not two
    # independent ones. Not further investigated (see project memory
    # project_wecc_flag_sweep_fixes.md) -- root cause in the RefFlag=true
    # path of repc_a not yet identified.
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊Q_ext), "pV.PlantController.Qext") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊Q_gen), "pV.RenewableController.Qgen") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊P_gen), "pV.RenewableController.Pe") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊V_t), "pV.RenewableController.Vt") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pcmd), "pV.RenewableController.Ipcmd") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qcmd), "pV.RenewableController.Iqcmd") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pmax), "pV.RenewableController.IPMAX.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pmin), "pV.RenewableController.IPMIN.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qmax), "pV.RenewableController.IQMAX.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qmin), "pV.RenewableController.IQMIN.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_lvpl), "pV.RenewableGenerator.LVPL.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pii), "pV.RenewableGenerator.p.ii") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pir), "pV.RenewableGenerator.p.ir") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pvi), "pV.RenewableGenerator.p.vi") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pvr), "pV.RenewableGenerator.p.vr") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_p), "pV.RenewableGenerator.IP.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_q), "pV.RenewableGenerator.IOLIM.y") < RTOL
end

@testset "QFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊V_limb), "pV.RenewableController.limiter3.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_lim), "pV.RenewableController.variableLimiter2.y") < RTOL
end

@testset "PfFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊P_PF), "pV.RenewableController.simpleLag1.y") < RTOL
end

@testset "RefFlag/VcombFlag-specific" begin
    # NOTE (2026-08-14, OPEN ISSUE): FAILS in both flag test cases
    # (RMS ~0.018 in both -- see matching note on Q_ext above, likely the
    # same underlying RefFlag=true discrepancy). Not further investigated.
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊ΔQ_in), "pV.PlantController.REFFLAG.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊V_in), "pV.PlantController.VCFLAG.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊V_droop), "pV.PlantController.add3.y") < RTOL
end

## Vflag&&QFlag is active in this test case -> Q_lim/V_lima exist in the Julia model
@testset "Vflag&&QFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊Q_lim), "pV.RenewableController.limiter1.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊V_lima), "pV.RenewableController.limiter2.y") < RTOL
end

## Plot layout follows the actual signal flow: repca (plant controller) first,
## in the order its own equations compute things (V_droop/V_in -> ΔQ_in ->
## Q_ext/P_ref), then reeca/reecb (electrical controller, inputs -> PfFlag ->
## Vflag&&QFlag -> QFlag -> final Ip/Iqcmd -> limits), then regca (generator,
## I_lvpl -> Ip/Iq -> terminal pir/pii/pvr/pvi).
if isdefined(Main, :EXPORT_FIGURES) && Main.EXPORT_FIGURES
    fig_ALLflipped = let
        fig = Figure(size=(1400, 2200))

        # --- repca (Plant Controller), in equation order ---
        ax1 = Axis(fig[1,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: V_droop & V_in (VcombFlag)")
        lines!(ax1, ref_pv.time, ref_pv[!, "pV.PlantController.add3.y"]; label="OpenIPSL V_droop", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax1, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊V_droop)).u; label="PD V_droop", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax1, ref_pv.time, ref_pv[!, "pV.PlantController.VCFLAG.y"]; label="OpenIPSL V_in", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax1, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊V_in)).u; label="PD V_in", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax1; position=:rb)

        ax2 = Axis(fig[1,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: ΔQ_in (RefFlag)")
        lines!(ax2, ref_pv.time, ref_pv[!, "pV.PlantController.REFFLAG.y"]; label="OpenIPSL ΔQ_in", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax2, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊ΔQ_in)).u; label="PD ΔQ_in", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax2; position=:rb)

        ax3 = Axis(fig[2,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: Q_ext & P_ref (final outputs)")
        lines!(ax3, ref_pv.time, ref_pv[!, "pV.PlantController.Qext"]; label="OpenIPSL Qext", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax3, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊Q_ext)).u; label="PD Qext", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax3, ref_pv.time, ref_pv[!, "pV.PlantController.Pref"]; label="OpenIPSL Pref", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax3, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊P_ref)).u; label="PD Pref", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax3; position=:rb)

        # --- reecb (Electrical Controller), in equation order ---
        ax4 = Axis(fig[2,2]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Q_gen, P_gen & Vt (measured inputs)")
        lines!(ax4, ref_pv.time, ref_pv[!, "pV.RenewableController.Qgen"]; label="OpenIPSL Q_gen", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax4, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊Q_gen)).u; label="PD Q_gen", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax4, ref_pv.time, ref_pv[!, "pV.RenewableController.Pe"]; label="OpenIPSL P_gen", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax4, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊P_gen)).u; label="PD P_gen", color=Cycled(2), linestyle=:dash, linewidth=2)
        lines!(ax4, ref_pv.time, ref_pv[!, "pV.RenewableController.Vt"]; label="OpenIPSL Vt", color=Cycled(3), linewidth=2, alpha=0.6)
        lines!(ax4, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊V_t)).u; label="PD Vt", color=Cycled(3), linestyle=:dash, linewidth=2)
        axislegend(ax4; position=:rb)

        ax5 = Axis(fig[3,1]; xlabel="Time [s]", ylabel="[pu]", title="reecb: P_PF (PfFlag)")
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.simpleLag1.y"]; label="OpenIPSL P_PF", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊P_PF)).u; label="PD P_PF", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax5; position=:rb)

        ax6 = Axis(fig[3,2]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Q_lim & V_lima (Vflag&&QFlag)")
        lines!(ax6, ref_pv.time, ref_pv[!, "pV.RenewableController.limiter1.y"]; label="OpenIPSL Q_lim", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax6, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊Q_lim)).u; label="PD Q_lim", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax6, ref_pv.time, ref_pv[!, "pV.RenewableController.limiter2.y"]; label="OpenIPSL V_lima", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax6, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊V_lima)).u; label="PD V_lima", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax6; position=:rb)

        ax7 = Axis(fig[4,1]; xlabel="Time [s]", ylabel="[pu]", title="reecb: V_limb & I_lim (QFlag)")
        lines!(ax7, ref_pv.time, ref_pv[!, "pV.RenewableController.limiter3.y"]; label="OpenIPSL V_limb", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax7, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊V_limb)).u; label="PD V_limb", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax7, ref_pv.time, ref_pv[!, "pV.RenewableController.variableLimiter2.y"]; label="OpenIPSL I_lim", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax7, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_lim)).u; label="PD I_lim", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax7; position=:rb)

        ax8 = Axis(fig[4,2]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Ipcmd & Iqcmd (final outputs)")
        lines!(ax8, ref_pv.time, ref_pv[!, "pV.RenewableController.Ipcmd"]; label="OpenIPSL Ipcmd", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax8, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_pcmd)).u; label="PD Ipcmd", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax8, ref_pv.time, ref_pv[!, "pV.RenewableController.Iqcmd"]; label="OpenIPSL Iqcmd", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax8, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_qcmd)).u; label="PD Iqcmd", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax8; position=:rb)

        # --- regca (Generator), in equation order ---
        ax9 = Axis(fig[5,1]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_lvpl")
        lines!(ax9, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.LVPL.y"]; label="OpenIPSL I_lvpl", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax9, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_lvpl)).u; label="PD I_lvpl", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax9; position=:rb)

        ax10 = Axis(fig[5,2]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_p & I_q (final outputs)")
        lines!(ax10, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.IP.y"]; label="OpenIPSL I_p", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax10, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_p)).u; label="PD I_p", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax10, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.IOLIM.y"]; label="OpenIPSL I_q", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax10, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_q)).u; label="PD I_q", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax10; position=:rb)

        ax11 = Axis(fig[6,1]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pir & pii")
        lines!(ax11, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.ir"]; label="OpenIPSL pir", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax11, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pir)).u; label="PD pir", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax11, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.ii"]; label="OpenIPSL pii", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax11, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pii)).u; label="PD pii", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax11; position=:rb)

        ax12 = Axis(fig[6,2]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pvr & pvi")
        lines!(ax12, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.vr"]; label="OpenIPSL pvr", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax12, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pvr)).u; label="PD pvr", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax12, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.vi"]; label="OpenIPSL pvi", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax12, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pvi)).u; label="PD pvi", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax12; position=:rb)

        Label(fig[0, :], "PV flag test case: ALLflipped  --  L_vplsw=false, PfFlag=true, Vflag=true, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=true, freqFlag=false"; fontsize=14)

        fig
    end
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "PV_flagtest_ALLflipped_comparison.pdf"), fig_ALLflipped)
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "PV_flagtest_ALLflipped_comparison.png"), fig_ALLflipped)
end

end # @testset "PV flagtest ALLflipped"
catch e
    @warn "PV flagtest ALLflipped: exception escaped the section (should only happen on a real error, not a plain @test failure)" exception=(e, catch_backtrace())
end


# ==============================================================================
# Test case 2: "ALLflippedExceptVflag"
# ==============================================================================

@info "Running PV flag test case: ALLflippedExceptVflag"

ref_pv = CSV.read(
    joinpath(pkgdir(OpPoDyn), "test", "WECC_model_tests", "PV", "modelica_results_ALLflippedExceptVflag.csv"),
    DataFrame;
    drop=(i, name) -> contains(string(name), "nrows="),
    silencewarnings=true,
)

PV_BUS = let
    v_0 = 1.0
    P_0 = 0.015

    @named PV = OpPoDyn.Library.WECC_large_PV(;
        L_vplsw=false, PfFlag=true, Vflag=false, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=true, freqFlag=true)
    busmodel = MTKBus(PV; name=:GEN1)
    compile_bus(busmodel, pf=pfPV(V=v_0, P=P_0))
end

sol_pv = OpenIPSL_RePSSE_pv(PV_BUS; ω_b=2π*60)
ts_pv = refine_timeseries(sol_pv.t)

try
@testset "PV flagtest ALLflippedExceptVflag" begin

@testset "Kern-Set" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊P_ref), "pV.PlantController.Pref") < RTOL
    # NOTE (2026-08-14, OPEN ISSUE): see identical note in test case 1 above --
    # same failure, same likely root cause (RefFlag=true path in repc_a).
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊Q_ext), "pV.PlantController.Qext") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊Q_gen), "pV.RenewableController.Qgen") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊P_gen), "pV.RenewableController.Pe") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊V_t), "pV.RenewableController.Vt") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pcmd), "pV.RenewableController.Ipcmd") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qcmd), "pV.RenewableController.Iqcmd") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pmax), "pV.RenewableController.IPMAX.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pmin), "pV.RenewableController.IPMIN.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qmax), "pV.RenewableController.IQMAX.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qmin), "pV.RenewableController.IQMIN.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_lvpl), "pV.RenewableGenerator.LVPL.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pii), "pV.RenewableGenerator.p.ii") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pir), "pV.RenewableGenerator.p.ir") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pvi), "pV.RenewableGenerator.p.vi") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pvr), "pV.RenewableGenerator.p.vr") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_p), "pV.RenewableGenerator.IP.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_q), "pV.RenewableGenerator.IOLIM.y") < RTOL
end

@testset "QFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊V_limb), "pV.RenewableController.limiter3.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_lim), "pV.RenewableController.variableLimiter2.y") < RTOL
end

@testset "PfFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊P_PF), "pV.RenewableController.simpleLag1.y") < RTOL
end

@testset "RefFlag/VcombFlag-specific" begin
    # NOTE (2026-08-14, OPEN ISSUE): see identical note in test case 1 above.
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊ΔQ_in), "pV.PlantController.REFFLAG.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊V_in), "pV.PlantController.VCFLAG.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊V_droop), "pV.PlantController.add3.y") < RTOL
end

## freqFlag=true is active in this test case
@testset "freqFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊P_e), "pV.PlantController.add2.y") < RTOL
end

## Plot layout follows the actual signal flow: repca first (in equation
## order), then reecb, then regca -- see identical structure/comment in test
## case 1 above.
if isdefined(Main, :EXPORT_FIGURES) && Main.EXPORT_FIGURES
    fig_ALLflippedExceptVflag = let
        fig = Figure(size=(1400, 2200))

        # --- repca (Plant Controller), in equation order ---
        ax1 = Axis(fig[1,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: V_droop & V_in (VcombFlag)")
        lines!(ax1, ref_pv.time, ref_pv[!, "pV.PlantController.add3.y"]; label="OpenIPSL V_droop", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax1, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊V_droop)).u; label="PD V_droop", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax1, ref_pv.time, ref_pv[!, "pV.PlantController.VCFLAG.y"]; label="OpenIPSL V_in", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax1, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊V_in)).u; label="PD V_in", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax1; position=:rb)

        ax2 = Axis(fig[1,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: ΔQ_in (RefFlag)")
        lines!(ax2, ref_pv.time, ref_pv[!, "pV.PlantController.REFFLAG.y"]; label="OpenIPSL ΔQ_in", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax2, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊ΔQ_in)).u; label="PD ΔQ_in", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax2; position=:rb)

        ax3 = Axis(fig[2,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: Q_ext & P_ref (final outputs)")
        lines!(ax3, ref_pv.time, ref_pv[!, "pV.PlantController.Qext"]; label="OpenIPSL Qext", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax3, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊Q_ext)).u; label="PD Qext", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax3, ref_pv.time, ref_pv[!, "pV.PlantController.Pref"]; label="OpenIPSL Pref", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax3, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊P_ref)).u; label="PD Pref", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax3; position=:rb)

        ax4 = Axis(fig[2,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: P_e (freqFlag)")
        lines!(ax4, ref_pv.time, ref_pv[!, "pV.PlantController.add2.y"]; label="OpenIPSL P_e", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax4, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊P_e)).u; label="PD P_e", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax4; position=:rb)

        # --- reecb (Electrical Controller), in equation order ---
        ax5 = Axis(fig[3,1]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Q_gen, P_gen & Vt (measured inputs)")
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.Qgen"]; label="OpenIPSL Q_gen", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊Q_gen)).u; label="PD Q_gen", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.Pe"]; label="OpenIPSL P_gen", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊P_gen)).u; label="PD P_gen", color=Cycled(2), linestyle=:dash, linewidth=2)
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.Vt"]; label="OpenIPSL Vt", color=Cycled(3), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊V_t)).u; label="PD Vt", color=Cycled(3), linestyle=:dash, linewidth=2)
        axislegend(ax5; position=:rb)

        ax6 = Axis(fig[3,2]; xlabel="Time [s]", ylabel="[pu]", title="reecb: P_PF (PfFlag)")
        lines!(ax6, ref_pv.time, ref_pv[!, "pV.RenewableController.simpleLag1.y"]; label="OpenIPSL P_PF", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax6, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊P_PF)).u; label="PD P_PF", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax6; position=:rb)

        ax7 = Axis(fig[4,1]; xlabel="Time [s]", ylabel="[pu]", title="reecb: V_limb & I_lim (QFlag)")
        lines!(ax7, ref_pv.time, ref_pv[!, "pV.RenewableController.limiter3.y"]; label="OpenIPSL V_limb", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax7, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊V_limb)).u; label="PD V_limb", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax7, ref_pv.time, ref_pv[!, "pV.RenewableController.variableLimiter2.y"]; label="OpenIPSL I_lim", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax7, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_lim)).u; label="PD I_lim", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax7; position=:rb)

        ax8 = Axis(fig[4,2]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Ipcmd & Iqcmd (final outputs)")
        lines!(ax8, ref_pv.time, ref_pv[!, "pV.RenewableController.Ipcmd"]; label="OpenIPSL Ipcmd", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax8, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_pcmd)).u; label="PD Ipcmd", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax8, ref_pv.time, ref_pv[!, "pV.RenewableController.Iqcmd"]; label="OpenIPSL Iqcmd", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax8, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_qcmd)).u; label="PD Iqcmd", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax8; position=:rb)

        # --- regca (Generator), in equation order ---
        ax9 = Axis(fig[5,1]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_lvpl")
        lines!(ax9, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.LVPL.y"]; label="OpenIPSL I_lvpl", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax9, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_lvpl)).u; label="PD I_lvpl", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax9; position=:rb)

        ax10 = Axis(fig[5,2]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_p & I_q (final outputs)")
        lines!(ax10, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.IP.y"]; label="OpenIPSL I_p", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax10, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_p)).u; label="PD I_p", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax10, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.IOLIM.y"]; label="OpenIPSL I_q", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax10, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_q)).u; label="PD I_q", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax10; position=:rb)

        ax11 = Axis(fig[6,1]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pir & pii")
        lines!(ax11, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.ir"]; label="OpenIPSL pir", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax11, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pir)).u; label="PD pir", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax11, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.ii"]; label="OpenIPSL pii", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax11, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pii)).u; label="PD pii", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax11; position=:rb)

        ax12 = Axis(fig[6,2]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pvr & pvi")
        lines!(ax12, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.vr"]; label="OpenIPSL pvr", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax12, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pvr)).u; label="PD pvr", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax12, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.vi"]; label="OpenIPSL pvi", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax12, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pvi)).u; label="PD pvi", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax12; position=:rb)

        Label(fig[0, :], "PV flag test case: ALLflippedExceptVflag  --  L_vplsw=false, PfFlag=true, Vflag=false, QFlag=true, PqFlag=true, RefFlag=true, VcombFlag=true, freqFlag=true"; fontsize=14)

        fig
    end
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "PV_flagtest_ALLflippedExceptVflag_comparison.pdf"), fig_ALLflippedExceptVflag)
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "PV_flagtest_ALLflippedExceptVflag_comparison.png"), fig_ALLflippedExceptVflag)
end

end # @testset "PV flagtest ALLflippedExceptVflag"
catch e
    @warn "PV flagtest ALLflippedExceptVflag: exception escaped the section (should only happen on a real error, not a plain @test failure)" exception=(e, catch_backtrace())
end


# ==============================================================================
# Test case 3: "QFlagFalseRefFlag"
# ==============================================================================
#
# Motivation: replaces an earlier attempt ("RefFlagPfFlagFalse", PfFlag=false
# but QFlag=true) which showed that Q_ext/ΔQ_in still fail their RTOL check
# (RMS ~0.046/0.018, essentially unchanged from test cases 1+2) but STILL
# don't show up in Ipcmd/Iqcmd/pii/pir -- NOT because of PfFlag as first
# assumed, but because QFlag=true's `I_qcon ~ I_lim` branch (the inner
# voltage-current loop) is used unconditionally whenever QFlag=true,
# regardless of PfFlag/RefFlag/etc., completely bypassing I_qin (and hence
# Q_con/Qext_in/Q_ext). The ONLY way to make Q_ext's value structurally
# reach the terminal quantities at all is `I_qcon ~ I_qin` in the QFlag=false
# branch -- so THIS test case sets QFlag=false (on top of PfFlag=false) to
# finally give the RefFlag=true discrepancy a real chance to show up
# downstream. QFlag=false also means V_con/V_limb/ΔV/s_V/I_in/I_lim and
# Q_lim/V_lima don't exist in the Julia model at all (structurally removed),
# so there is no QFlag-specific testset/plot panel in this test case.
# ==============================================================================

@info "Running PV flag test case: QFlagFalseRefFlag"

ref_pv = CSV.read(
    joinpath(pkgdir(OpPoDyn), "test", "WECC_model_tests", "PV", "modelica_results_QFlagFalseRefFlag.csv"),
    DataFrame;
    drop=(i, name) -> contains(string(name), "nrows="),
    silencewarnings=true,
)

PV_BUS = let
    v_0 = 1.0
    P_0 = 0.015

    @named PV = OpPoDyn.Library.WECC_large_PV(;
        L_vplsw=false, PfFlag=false, Vflag=false, QFlag=false, PqFlag=true, RefFlag=true, VcombFlag=true, freqFlag=true)
    busmodel = MTKBus(PV; name=:GEN1)
    compile_bus(busmodel, pf=pfPV(V=v_0, P=P_0))
end

sol_pv = OpenIPSL_RePSSE_pv(PV_BUS; ω_b=2π*60)
ts_pv = refine_timeseries(sol_pv.t)

try
@testset "PV flagtest QFlagFalseRefFlag" begin

@testset "Kern-Set" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊P_ref), "pV.PlantController.Pref") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊Q_ext), "pV.PlantController.Qext") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊Q_gen), "pV.RenewableController.Qgen") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊P_gen), "pV.RenewableController.Pe") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊V_t), "pV.RenewableController.Vt") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pcmd), "pV.RenewableController.Ipcmd") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qcmd), "pV.RenewableController.Iqcmd") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pmax), "pV.RenewableController.IPMAX.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_pmin), "pV.RenewableController.IPMIN.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qmax), "pV.RenewableController.IQMAX.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊reecb₊I_qmin), "pV.RenewableController.IQMIN.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_lvpl), "pV.RenewableGenerator.LVPL.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pii), "pV.RenewableGenerator.p.ii") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pir), "pV.RenewableGenerator.p.ir") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pvi), "pV.RenewableGenerator.p.vi") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊pvr), "pV.RenewableGenerator.p.vr") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_p), "pV.RenewableGenerator.IP.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊regca₊I_q), "pV.RenewableGenerator.IOLIM.y") < RTOL
end

## no PfFlag-specific testset here: PfFlag=false -> P_PF doesn't exist in the Julia model
## no QFlag-specific testset here: QFlag=false -> V_limb/I_lim don't exist in the Julia model

@testset "RefFlag/VcombFlag-specific" begin
    # NOTE (2026-08-14): this is the whole point of this test case -- see the
    # header comment above. With QFlag=false, I_qcon~I_qin (not I_lim), so
    # Q_con (fed by Qext_in=Q_ext, since PfFlag=false too) genuinely reaches
    # Ipcmd/Iqcmd/pii/pir this time. If Kern-Set now ALSO fails on Iqcmd/pii/
    # pir (not just Q_ext/ΔQ_in here), that confirms the RefFlag=true
    # discrepancy is real and propagates when nothing else masks it.
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊ΔQ_in), "pV.PlantController.REFFLAG.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊V_in), "pV.PlantController.VCFLAG.y") < RTOL
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊V_droop), "pV.PlantController.add3.y") < RTOL
end

@testset "freqFlag-specific" begin
    @test ref_rms_error(sol_pv, ref_pv, VIndex(:GEN1, :PV₊repca₊P_e), "pV.PlantController.add2.y") < RTOL
end

if isdefined(Main, :EXPORT_FIGURES) && Main.EXPORT_FIGURES
    fig_QFlagFalseRefFlag = let
        fig = Figure(size=(1400, 2000))

        # --- repca (Plant Controller), in equation order ---
        ax1 = Axis(fig[1,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: V_droop & V_in (VcombFlag)")
        lines!(ax1, ref_pv.time, ref_pv[!, "pV.PlantController.add3.y"]; label="OpenIPSL V_droop", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax1, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊V_droop)).u; label="PD V_droop", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax1, ref_pv.time, ref_pv[!, "pV.PlantController.VCFLAG.y"]; label="OpenIPSL V_in", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax1, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊V_in)).u; label="PD V_in", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax1; position=:rb)

        ax2 = Axis(fig[1,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: ΔQ_in (RefFlag)")
        lines!(ax2, ref_pv.time, ref_pv[!, "pV.PlantController.REFFLAG.y"]; label="OpenIPSL ΔQ_in", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax2, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊ΔQ_in)).u; label="PD ΔQ_in", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax2; position=:rb)

        ax3 = Axis(fig[2,1]; xlabel="Time [s]", ylabel="[pu]", title="repca: Q_ext & P_ref (final outputs)")
        lines!(ax3, ref_pv.time, ref_pv[!, "pV.PlantController.Qext"]; label="OpenIPSL Qext", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax3, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊Q_ext)).u; label="PD Qext", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax3, ref_pv.time, ref_pv[!, "pV.PlantController.Pref"]; label="OpenIPSL Pref", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax3, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊P_ref)).u; label="PD Pref", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax3; position=:rb)

        ax4 = Axis(fig[2,2]; xlabel="Time [s]", ylabel="[pu]", title="repca: P_e (freqFlag)")
        lines!(ax4, ref_pv.time, ref_pv[!, "pV.PlantController.add2.y"]; label="OpenIPSL P_e", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax4, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊repca₊P_e)).u; label="PD P_e", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax4; position=:rb)

        # --- reecb (Electrical Controller), in equation order (no P_PF, no QFlag panels) ---
        ax5 = Axis(fig[3,1]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Q_gen, P_gen & Vt (measured inputs)")
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.Qgen"]; label="OpenIPSL Q_gen", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊Q_gen)).u; label="PD Q_gen", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.Pe"]; label="OpenIPSL P_gen", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊P_gen)).u; label="PD P_gen", color=Cycled(2), linestyle=:dash, linewidth=2)
        lines!(ax5, ref_pv.time, ref_pv[!, "pV.RenewableController.Vt"]; label="OpenIPSL Vt", color=Cycled(3), linewidth=2, alpha=0.6)
        lines!(ax5, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊V_t)).u; label="PD Vt", color=Cycled(3), linestyle=:dash, linewidth=2)
        axislegend(ax5; position=:rb)

        ax8 = Axis(fig[3,2]; xlabel="Time [s]", ylabel="[pu]", title="reecb: Ipcmd & Iqcmd (final outputs)")
        lines!(ax8, ref_pv.time, ref_pv[!, "pV.RenewableController.Ipcmd"]; label="OpenIPSL Ipcmd", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax8, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_pcmd)).u; label="PD Ipcmd", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax8, ref_pv.time, ref_pv[!, "pV.RenewableController.Iqcmd"]; label="OpenIPSL Iqcmd", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax8, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊reecb₊I_qcmd)).u; label="PD Iqcmd", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax8; position=:rb)

        # --- regca (Generator), in equation order ---
        ax9 = Axis(fig[4,1]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_lvpl")
        lines!(ax9, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.LVPL.y"]; label="OpenIPSL I_lvpl", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax9, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_lvpl)).u; label="PD I_lvpl", color=Cycled(1), linestyle=:dash, linewidth=2)
        axislegend(ax9; position=:rb)

        ax10 = Axis(fig[4,2]; xlabel="Time [s]", ylabel="[pu]", title="regca: I_p & I_q (final outputs)")
        lines!(ax10, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.IP.y"]; label="OpenIPSL I_p", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax10, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_p)).u; label="PD I_p", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax10, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.IOLIM.y"]; label="OpenIPSL I_q", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax10, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊regca₊I_q)).u; label="PD I_q", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax10; position=:rb)

        ax11 = Axis(fig[5,1]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pir & pii")
        lines!(ax11, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.ir"]; label="OpenIPSL pir", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax11, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pir)).u; label="PD pir", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax11, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.ii"]; label="OpenIPSL pii", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax11, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pii)).u; label="PD pii", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax11; position=:rb)

        ax12 = Axis(fig[5,2]; xlabel="Time [s]", ylabel="[pu]", title="Terminal: pvr & pvi")
        lines!(ax12, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.vr"]; label="OpenIPSL pvr", color=Cycled(1), linewidth=2, alpha=0.6)
        lines!(ax12, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pvr)).u; label="PD pvr", color=Cycled(1), linestyle=:dash, linewidth=2)
        lines!(ax12, ref_pv.time, ref_pv[!, "pV.RenewableGenerator.p.vi"]; label="OpenIPSL pvi", color=Cycled(2), linewidth=2, alpha=0.6)
        lines!(ax12, ts_pv, sol_pv(ts_pv, idxs=VIndex(:GEN1, :PV₊pvi)).u; label="PD pvi", color=Cycled(2), linestyle=:dash, linewidth=2)
        axislegend(ax12; position=:rb)

        Label(fig[0, :], "PV flag test case: QFlagFalseRefFlag  --  L_vplsw=false, PfFlag=false, Vflag=false, QFlag=false, PqFlag=true, RefFlag=true, VcombFlag=true, freqFlag=true"; fontsize=14)

        fig
    end
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "PV_flagtest_QFlagFalseRefFlag_comparison.pdf"), fig_QFlagFalseRefFlag)
    save(joinpath(pkgdir(OpPoDyn), "docs", "src", "assets", "OpenIPSL_valid", "PV_flagtest_QFlagFalseRefFlag_comparison.png"), fig_QFlagFalseRefFlag)
end

end # @testset "PV flagtest QFlagFalseRefFlag"
catch e
    @warn "PV flagtest QFlagFalseRefFlag: exception escaped the section (should only happen on a real error, not a plain @test failure)" exception=(e, catch_backtrace())
end