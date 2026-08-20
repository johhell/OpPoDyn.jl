@mtkmodel reec_b begin
    @structural_parameters begin
        PfFlag = false
        Vflag = false
        QFlag = false
        PqFlag = false
    end
    @parameters begin
        V_dip, [description="Low voltage condition trigger voltage (pu)"]
        V_up, [description="High voltage condition trigger voltage (pu)"]
        T_rv, [description="Terminal bus voltage filter time constant (s)"]
        V_ref0, [description="Reference voltage for reactive current injection (pu)"]
        dbd1, [description="Overvoltage deadband for reactive current injection (pu)"]
        dbd2, [description="Undervoltage deadband for reactive current injection (pu)"]
        K_qv, [description="Reactive current injection gain (pu/pu)"]
        I_qh1, [description="Maximum reactive current injection (pu on mbase)"]
        I_ql1, [description="Minimum reactive current injection (pu on mbase)"]
        T_p, [description="Active power filter time constant (s)"]
        Q_min, [description="Minimum reactive power when Vflag = 1 (pu on mbase)"]
        Q_max, [description="Maximum reactive power when Vflag = 1 (pu on mbase)"]
        V_min, [description="Minimum voltage at inverter terminal bus (pu)"]
        V_max, [description="Maximum voltage at inverter terminal bus (pu)"]
        K_qp, [description="Local Q regulator proportional gain (pu/pu)"]
        K_qi, [description="Local Q regulator integral gain (pu/pu-s)"]
        K_vp, [description="Local voltage regulator proportional gain (pu/pu)"]
        K_vi, [description="Local voltage regulator integral gain (pu/pu-s)"]
        I_max, [description="Maximum apparent current (pu on mbase)"]
        T_iq, [description="Reactive current regulator lag time constant (s)"]
        T_pord, [description="Inverter power order lag time constant (s)"]
        P_min, [description="Minimum active power (pu on mbase)"]
        P_max, [description="Maximum active power (pu on mbase)"]
        dP_min, [description="Active power down-ramp limit (pu/s on mbase)"]
        dP_max, [description="Active power up-ramp limit (pu/s on mbase)"]
    end
    @components begin
        Vt_in = RealInput(guess=1)
        P_e = RealInput(guess=0.015)
        P_faref = RealInput(guess=-1.31199)
        Qext_in = RealInput(guess=-0.056656797)
        Pref_in = RealInput(guess=0.015)
        Q_gen = RealInput(guess=-0.056656801)
        # outputs
        Iqcmd_out = RealOutput(guess=-0.056656797)
        Ipcmd_out = RealOutput(guess=0.015)

        #building blocks
        simpleLag = PowerDynamics.Library.SimpleLag(K=1, T=T_rv, guess=1)
        deadband = PowerDynamics.Library.DeadZone(uMax=dbd2, uMin=dbd1)
        if PfFlag
            simpleLag1 = PowerDynamics.Library.SimpleLag(K=1, T=T_p, guess=0.015)
        end
    end
    @variables begin
        Voltage_dip(t), [guess=0, description="freeze states if Voltagedip=1"]
        V_tfilt(t), [guess=1, description="Voltage after filter"]
        V_tfiltlim(t), [guess=1, description="Voltage after filter with lower limit 0.01"]
        ΔV_t(t), [guess=6.6613381e-14, description="Difference between filterd terminal voltage and reference voltage"]
        ΔV_tdbd(t), [guess=0, description="Voltage after deadband"]
        I_qinj(t), [guess=0, description="Limited current injection q-Phase from Voltage"]
        if PfFlag
            P_PF(t), [guess=0.015, description="Inverter active power after filter"]
        end
        Q_con(t), [guess=-0.056656797, description="Reactive Power after PfFlag"]
        if Vflag && QFlag
            Q_lim(t), [guess=-0.056656797, description="ReactivePower after limiter"]
            ΔQ(t), [guess=4.8269116e-9, description="Difference between Q_lim and Q_gen"]
            s_Q(t), [guess=4.8269116e-9, description="Frozen state in Q regulator"]
            s_Qint(t), [guess=1, description=""]
            V_in(t), [guess=1, description="Voltage after local Q regulator"]
            V_lima(t), [guess=1, description="Limited voltage after Q regulator"]
        end
        if QFlag
            V_con(t), [guess=1, description="Voltage after Vflag"]
            V_limb(t), [guess=1, description="Limited voltage V_con"]
            ΔV(t), [guess=6.6613381e-14, description="Difference between V_limb and V_tfilt"]
            s_V(t), [guess=6.6613381e-14, description="Frozen state at local voltage regulator"]
            s_Vint(t), [guess=-0.0567, description=""]
            I_in(t), [guess=-0.0567, description="Current after local voltage regulator"]
            I_lim(t), [guess=-0.0567, description="limited current after voltage regulator"]
        end
        I_t(t), [guess=-0.056656797, description="Current from Q_con/V_tfiltlim"]
        ΔI(t), [guess=-1.9310942e-14, description=""]
        I_qin(t), [guess=-0.056656797, description="Current after Reactive current regulator"]
        I_qcon(t), [guess=-0.056656797, description="Current after QFlag"]
        I_sum(t), [guess=-0.056656797, description="sum of I_qcon and I_qinj"]
        I_qcmd(t), [guess=-0.056656797, description="q-Phase output current"]
        P_refout(t), [guess=0.015, description="Active power after inverter power order"]
        P_lim(t), [guess=0.015, description="Limited active power after inverter power order"]
        ΔP(t), [guess=-7.3598558e-12, description="Active power difference between P_ref and P_refout"]
        ΔP_lim(t), [guess=-7.3598558e-12, description="Ramp-limited active power difference"]
        I_pref(t), [guess=0.015, description="Current from P_lim/V_tfiltlim"]
        I_pcmd(t), [guess=0.015, description="p-Phase output current"]
        I_qmin(t), [guess=-1.82, description="Minumum q-Phase current limit (pu)"]
        I_qmax(t), [guess=1.82, description="Maximum q-Phase current limit (pu)"]
        I_pmax(t), [guess=1.8191179, description="Maximum p-Phase current limit (pu)"]
        I_pmin(t), [guess=0, description="Minumum p-Phase current limit (pu)"]
    end
    @equations begin
        Voltage_dip ~ ifelse(Vt_in.u<V_dip, 1, ifelse(Vt_in.u>V_up, 1, 0))

        simpleLag.in ~ Vt_in.u
        V_tfilt ~ simpleLag.out

        V_tfiltlim ~ max(V_tfilt, 0.01)
        ΔV_t ~ V_ref0 - V_tfilt

        deadband.in ~ ΔV_t
        ΔV_tdbd ~ deadband.out

        I_qinj ~ clamp(K_qv*ΔV_tdbd, I_ql1, I_qh1)

        if PfFlag
            simpleLag1.in ~ P_e.u
            P_PF ~ simpleLag1.out
            Q_con ~ P_PF * tan(P_faref.u)
        else
            Q_con ~ Qext_in.u
        end

        if Vflag && QFlag
            Q_lim ~ clamp(Q_con, Q_min, Q_max)
            ΔQ ~ Q_lim - Q_gen.u
            s_Q ~ (1-Voltage_dip) * ΔQ
            Dt(s_Qint) ~ K_qi * s_Q
            V_in ~ K_qp * s_Q + s_Qint
            V_lima ~ clamp(V_in, V_min, V_max)
            V_con ~ V_lima
        end
        if !Vflag && QFlag
            V_con ~ V_ref0
        end
        if QFlag
            V_limb ~ clamp(V_con, V_min, V_max)
            ΔV ~ V_limb - V_tfilt
            s_V ~ (1-Voltage_dip) * ΔV
            Dt(s_Vint) ~ K_vi * s_V
            I_in ~ K_vp * s_V + s_Vint
            I_lim ~ clamp(I_in, I_qmin, I_qmax)
            I_qcon ~ I_lim
        else
            I_qcon ~ I_qin
        end

        I_t ~ Q_con / V_tfiltlim
        ΔI ~ I_t - I_qin
        T_iq * Dt(I_qin) ~ (1-Voltage_dip) * ΔI
        I_sum ~ I_qcon + I_qinj
        I_qcmd ~ clamp(I_sum, I_qmin, I_qmax)
        #p-phase current
        ΔP ~ Pref_in.u - P_refout
        ΔP_lim ~ clamp(ΔP, dP_min, dP_max)
        T_pord * Dt(P_refout) ~ (1-Voltage_dip) * ΔP_lim
        P_lim ~ clamp(P_refout, P_min, P_max)
        I_pref ~ P_lim/V_tfiltlim
        I_pcmd ~ clamp(I_pref, I_pmin, I_pmax)
        #current limiter logic
        I_pmin ~ 0
        I_qmin ~ - I_qmax
        I_pmax ~ ifelse(PqFlag, I_max, sqrt(I_max^2 - I_qcmd^2))
        I_qmax ~ ifelse(PqFlag, sqrt(I_max^2 - I_pcmd^2), I_max)
        #outputs
        Iqcmd_out.u ~ I_qcmd
        Ipcmd_out.u ~ I_pcmd
    end
end



@mtkmodel reec_c begin
    @structural_parameters begin
        PfFlag = false
        Vflag = false
        QFlag = false
        PqFlag = false
    end
    @parameters begin
        V_dip, [description="Low voltage condition trigger voltage (pu)"]
        V_up, [description="High voltage condition trigger voltage (pu)"]
        T_rv, [description="Terminal bus voltage filter time constant (s)"]
        V_ref0, [description="Reference voltage for reactive current injection (pu)"]
        dbd1, [description="Overvoltage deadband for reactive current injection (pu)"]
        dbd2, [description="Undervoltage deadband for reactive current injection (pu)"]
        K_qv, [description="Reactive current injection gain (pu/pu)"]
        I_qh1, [description="Maximum reactive current injection (pu on mbase)"]
        I_ql1, [description="Minimum reactive current injection (pu on mbase)"]
        T_p, [description="Active power filter time constant (s)"]
        Q_min, [description="Minimum reactive power when Vflag = 1 (pu on mbase)"]
        Q_max, [description="Maximum reactive power when Vflag = 1 (pu on mbase)"]
        V_min, [description="Minimum voltage at inverter terminal bus (pu)"]
        V_max, [description="Maximum voltage at inverter terminal bus (pu)"]
        K_qp, [description="Local Q regulator proportional gain (pu/pu)"]
        K_qi, [description="Local Q regulator integral gain (pu/pu-s)"]
        K_vp, [description="Local voltage regulator proportional gain (pu/pu)"]
        K_vi, [description="Local voltage regulator integral gain (pu/pu-s)"]
        I_max, [description="Maximum apparent current (pu on mbase)"]
        T_iq, [description="Reactive current regulator lag time constant (s)"]
        T_pord, [description="Inverter power order lag time constant (s)"]
        P_min, [description="Minimum active power (pu on mbase)"]
        P_max, [description="Maximum active power (pu on mbase)"]
        dP_min, [description="Active power down-ramp limit (pu/s on mbase)"]
        dP_max, [description="Active power up-ramp limit (pu/s on mbase)"]
        soc_ini, [description="initial state of charge"]
        T_char, [description="Battery discharge time"]
        SOCmin, [description="Minimum allowable state of charge"]
        SOCmax, [description="Maximum allowable state of charge"]
        Vq1=0.0, [description="q-VDL Table"]
        Vq2=0.2, [description="q-VDL Table"]
        Vq3=0.5, [description="q-VDL Table"]
        Vq4=1, [description="q-VDL Table"]
        Iq1=0.75, [description="q-VDL Table"]
        Iq2=0.75, [description="q-VDL Table"]
        Iq3=0.75, [description="q-VDL Table"]
        Iq4=0.75, [description="q-VDL Table"]
        Vp1=0.2, [description="p-VDL Table"]
        Vp2=0.5, [description="p-VDL Table"]
        Vp3=0.75, [description="p-VDL Table"]
        Vp4=1, [description="p-VDL Table"]
        Ip1=1.11, [description="p-VDL Table"]
        Ip2=1.11, [description="p-VDL Table"]
        Ip3=1.11, [description="p-VDL Table"]
        Ip4=1.11, [description="p-VDL Table"]
    end
    @components begin
        Vt_in = RealInput(guess=1)
        P_e = RealInput(guess=0.015)
        P_faref = RealInput(guess=-1.31199)
        Qext_in = RealInput(guess=-0.056656797)
        Pref_in = RealInput(guess=0.015)
        Q_gen = RealInput(guess=-0.056656801)
        P_aux = RealInput(guess=0)
        PELEC = RealInput(guess=0.015)
        # outputs
        Iqcmd_out = RealOutput(guess=-0.056656797)
        Ipcmd_out = RealOutput(guess=0.015)

        #building blocks
        simpleLag = PowerDynamics.Library.SimpleLag(K=1, T=T_rv, guess=1)
        deadband = PowerDynamics.Library.DeadZone(uMax=dbd2, uMin=dbd1)
        if PfFlag
            simpleLag1 = PowerDynamics.Library.SimpleLag(K=1, T=T_p, guess=0.015)
        end
    end
    @variables begin
        Voltage_dip(t), [guess=0, description="freeze states if Voltagedip=1"]
        V_tfilt(t), [guess=1, description="Voltage after filter"]
        V_tfiltlim(t), [guess=1, description="Voltage after filter with lower limit 0.01"]
        ΔV_t(t), [guess=6.6613381e-14, description="Difference between filterd terminal voltage and reference voltage"]
        ΔV_tdbd(t), [guess=0, description="Voltage after deadband"]
        I_qinj(t), [guess=0, description="Limited current injection q-Phase from Voltage"]
        if PfFlag
            P_PF(t), [guess=0.015, description="Inverter active power after filter"]
        end
        Q_con(t), [guess=-0.056656797, description="Reactive Power after PfFlag"]
        if Vflag && QFlag
            Q_lim(t), [guess=-0.056656797, description="ReactivePower after limiter"]
            ΔQ(t), [guess=4.8269116e-9, description="Difference between Q_lim and Q_gen"]
            s_Q(t), [guess=4.8269116e-9, description="Frozen state in Q regulator"]
            s_Qint(t), [guess=1, description=""]
            V_in(t), [guess=1, description="Voltage after local Q regulator"]
            V_lima(t), [guess=1, description="Limited voltage after Q regulator"]
        end
        if QFlag
            V_con(t), [guess=1, description="Voltage after Vflag"]
            V_limb(t), [guess=1, description="Limited voltage V_con"]
            ΔV(t), [guess=6.6613381e-14, description="Difference between V_limb and V_tfilt"]
            s_V(t), [guess=6.6613381e-14, description="Frozen state at local voltage regulator"]
            s_Vint(t), [guess=-0.0567, description=""]
            I_in(t), [guess=-0.0567, description="Current after local voltage regulator"]
            I_lim(t), [guess=-0.0567, description="limited current after voltage regulator"]
        end
        I_t(t), [guess=-0.056656797, description="Current from Q_con/V_tfiltlim"]
        ΔI(t), [guess=0, description=""]
        I_qin(t), [guess=-0.056656797, description="Current after Reactive current regulator"]
        I_qcon(t), [guess=-0.056656797, description="Current after QFlag"]
        I_sum(t), [guess=-0.056656797, description="sum of I_qcon and I_qinj"]
        I_qcmd(t), [guess=-0.056656797, description="q-Phase output current"]
        P_refout(t), [guess=0.015, description="Active power after inverter power order"]
        P_lim(t), [guess=0.015, description="Limited active power after inverter power order"]
        ΔP(t), [guess=0, description="Active power difference between P_ref and P_refout"]
        ΔP_lim(t), [guess=0, description="Ramp-limited active power difference"]
        I_pref(t), [guess=0.015, description="Current from P_lim/V_tfiltlim"]
        ΔI_p(t), [guess=0.015, description=""]
        I_pcmd(t), [guess=0.015, description="p-Phase output current"]
        I_qmin(t), [guess=-0.75, description="Minumum q-Phase current limit (pu)"]
        I_qmax(t), [guess=0.75, description="Maximum q-Phase current limit (pu)"]
        I_pmax(t), [guess=1.108553114, description="Maximum p-Phase current limit (pu)"]
        I_pmin(t), [guess=-1.108553114, description="Minumum p-Phase current limit (pu)"]
        I_pmin_soc(t), [guess=-1.108553114, description=""]
        I_pmax_soc(t), [guess=1.108553114, description=""]
        soc_Imin(t), [guess=1, description=""]
        soc_Imax(t), [guess=1, description=""]
        P_stor(t), [guess=0.015, description=""]
        soc(t), [guess=0.485, description=""]
        soc_lim(t), [guess=0.485, description=""]
        VDL1_out(t), [guess=0.75, description=""]
        VDL2_out(t), [guess=1.11, description=""]
    end
    @equations begin
        Voltage_dip ~ ifelse(Vt_in.u<V_dip, 1, ifelse(Vt_in.u>V_up, 1, 0))

        simpleLag.in ~ Vt_in.u
        V_tfilt ~ simpleLag.out

        V_tfiltlim ~ max(V_tfilt, 0.01)
        ΔV_t ~ V_ref0 - V_tfilt

        deadband.in ~ ΔV_t
        ΔV_tdbd ~ deadband.out

        I_qinj ~ clamp(K_qv*ΔV_tdbd, I_ql1, I_qh1)

        if PfFlag
            simpleLag1.in ~ P_e.u
            P_PF ~ simpleLag1.out
            Q_con ~ P_PF * tan(P_faref.u)
        else
            Q_con ~ Qext_in.u
        end

        if Vflag && QFlag
            Q_lim ~ clamp(Q_con, Q_min, Q_max)
            ΔQ ~ Q_lim - Q_gen.u
            s_Q ~ (1-Voltage_dip) * ΔQ
            Dt(s_Qint) ~ K_qi * s_Q
            V_in ~ K_qp * s_Q + s_Qint
            V_lima ~ clamp(V_in, V_min, V_max)
            V_con ~ V_lima
        end
        if !Vflag && QFlag
            V_con ~ V_ref0
        end
        if QFlag
            V_limb ~ clamp(V_con, V_min, V_max)
            ΔV ~ V_limb - V_tfilt
            s_V ~ (1-Voltage_dip) * ΔV
            Dt(s_Vint) ~ K_vi * s_V
            I_in ~ K_vp * s_V + s_Vint
            I_lim ~ clamp(I_in, I_qmin, I_qmax)
            I_qcon ~ I_lim
        else
            I_qcon ~ I_qin
        end

        I_t ~ Q_con / V_tfiltlim
        ΔI ~ I_t - I_qin
        T_iq * Dt(I_qin) ~ (1-Voltage_dip) * ΔI
        I_sum ~ I_qcon + I_qinj
        I_qcmd ~ clamp(I_sum, I_qmin, I_qmax)
        #p-phase current
        ΔP ~ Pref_in.u - P_refout
        ΔP_lim ~ clamp(ΔP, dP_min, dP_max)
        T_pord * Dt(P_refout) ~ (1-Voltage_dip) * ΔP_lim
        P_lim ~ clamp(P_refout, P_min, P_max)
        I_pref ~ P_lim/V_tfiltlim
        ΔI_p ~ P_aux.u + I_pref
        I_pmin_soc ~ I_pmin * soc_Imin
        I_pmax_soc ~ I_pmax * soc_Imax
        I_pcmd ~ clamp(ΔI_p, I_pmin_soc, I_pmax_soc)
        #soc logic
        T_char * Dt(P_stor) ~ PELEC.u
        soc ~ soc_ini - P_stor
        soc_lim ~ clamp(soc, SOCmin, SOCmax)
        soc_Imax ~ ifelse(soc_lim<=SOCmin, 0, 1)
        soc_Imin ~ ifelse(soc_lim>=SOCmax, 0, 1)
        #VDL tables
        VDL1_out ~ VDL(V_tfilt, Vq1, Vq2, Vq3, Vq4, Iq1, Iq2, Iq3, Iq4)
        VDL2_out ~ VDL(V_tfilt, Vp1, Vp2, Vp3, Vp4, Ip1, Ip2, Ip3, Ip4)
        #current limiter logic
        I_pmin ~ -I_pmax
        I_qmin ~ -I_qmax
        I_pmax ~ ifelse(PqFlag, min(VDL2_out, I_max), min(VDL2_out, sqrt(I_max^2 - I_qcmd^2)))
        I_qmax ~ ifelse(PqFlag, min(VDL1_out, sqrt(I_max^2 - I_pcmd^2)), min(VDL1_out, I_max))
        #outputs
        Iqcmd_out.u ~ I_qcmd
        Ipcmd_out.u ~ I_pcmd
    end
end


@mtkmodel reec_a begin
    @structural_parameters begin
        PfFlag = false
        Vflag = false
        QFlag = false
        PqFlag = false
        # NOTE (2026-08-17): PFlag corresponds to OpenIPSL's `pflag`
        # (BaseREECA.mo), which is a SEPARATE parameter from `pfflag`. In
        # REECA1.mo it drives only
        #     GeneratorSpeed.y = if pflag then Wg else 1
        # i.e. whether the active power reference is scaled by the generator
        # speed. Until now this branch was (incorrectly) keyed on PfFlag here,
        # so the two independent Modelica flags were collapsed into one. That
        # stayed invisible for the validated baseline because WindPlant.mo
        # derives both as false there (QFunctionality=4 -> pfflag=false,
        # TOscillation=0 -> pflag=false), but it would silently diverge from
        # the reference in any case with pfflag=true and pflag=false.
        # Default false = previous behaviour for the default PfFlag=false, and
        # matches the reference model.
        PFlag = false
    end
    @parameters begin
        V_0, [description="Initial/nominal terminal voltage (pu); used as Vmod's V0 when !PfFlag && !Vflag && QFlag, see OpenIPSL REECA1.mo"]
        V_dip, [description="Low voltage condition trigger voltage (pu)"]
        V_up, [description="High voltage condition trigger voltage (pu)"]
        T_rv, [description="Terminal bus voltage filter time constant (s)"]
        V_ref0, [description="Reference voltage for reactive current injection (pu); has to be !=0"]
        dbd1, [description="Overvoltage deadband for reactive current injection (pu)"]
        dbd2, [description="Undervoltage deadband for reactive current injection (pu)"]
        K_qv, [description="Reactive current injection gain (pu/pu)"]
        I_qh1, [description="Maximum reactive current injection (pu on mbase)"]
        I_ql1, [description="Minimum reactive current injection (pu on mbase)"]
        T_p, [description="Active power filter time constant (s)"]
        Q_min, [description="Minimum reactive power when Vflag = 1 (pu on mbase)"]
        Q_max, [description="Maximum reactive power when Vflag = 1 (pu on mbase)"]
        V_min, [description="Minimum voltage at inverter terminal bus (pu)"]
        V_max, [description="Maximum voltage at inverter terminal bus (pu)"]
        K_qp, [description="Local Q regulator proportional gain (pu/pu)"]
        K_qi, [description="Local Q regulator integral gain (pu/pu-s)"]
        V_bias, [description="User-defined reference/bias on the inner-loop voltage control (pu); only used directly when PfFlag=true, otherwise overridden by Vmod, see OpenIPSL REECA1.mo"]
        K_vp, [description="Local voltage regulator proportional gain (pu/pu)"]
        K_vi, [description="Local voltage regulator integral gain (pu/pu-s)"]
        I_max, [description="Maximum apparent current (pu on mbase)"]
        T_iq, [description="Reactive current regulator lag time constant (s)"]
        T_pord, [description="Inverter power order lag time constant (s)"]
        P_min, [description="Minimum active power (pu on mbase)"]
        P_max, [description="Maximum active power (pu on mbase)"]
        dP_min, [description="Active power down-ramp limit (pu/s on mbase)"]
        dP_max, [description="Active power up-ramp limit (pu/s on mbase)"]
        Vq1=0.1, [description="q-VDL Table 1"]
        Vq2=0.4, [description="q-VDL Table 1"]
        Vq3=0.6, [description="q-VDL Table 1"]
        Vq4=0.9, [description="q-VDL Table 1"]
        Iq1=0.01, [description="q-VDL Table 1"]
        Iq2=0.5, [description="q-VDL Table 1"]
        Iq3=0.7, [description="q-VDL Table 1"]
        Iq4=1.0, [description="q-VDL Table 1"]
        Vp1=0.1, [description="p-VDL Table 2"]
        Vp2=0.5, [description="p-VDL Table 2"]
        Vp3=0.9, [description="p-VDL Table 2"]
        Vp4=1, [description="p-VDL Table 2"]
        Ip1=0.4, [description="p-VDL Table 2"]
        Ip2=0.7, [description="p-VDL Table 2"]
        Ip3=1.2, [description="p-VDL Table 2"]
        Ip4=1.2, [description="p-VDL Table 2"]
    end
    @components begin
        Vt_in = RealInput(guess=1)
        P_e = RealInput(guess=0.015)
        P_faref = RealInput(guess=-1.31199)
        Qext_in = RealInput(guess=-0.056656797)
        Pref_in = RealInput(guess=0.015)
        Q_gen = RealInput(guess=-0.056656801)
        Wg = RealInput(guess=1)
        # outputs
        Iqcmd_out = RealOutput(guess=-0.056656797)
        Ipcmd_out = RealOutput(guess=0.015)

        #building blocks
        simpleLag = PowerDynamics.Library.SimpleLag(K=1, T=T_rv, guess=1)
        deadband = PowerDynamics.Library.DeadZone(uMax=dbd2, uMin=dbd1)
        if PfFlag
            simpleLag1 = PowerDynamics.Library.SimpleLag(K=1, T=T_p, guess=0.015)
        end
    end
    @variables begin
        Voltage_dip(t), [guess=0, description="freeze states if Voltagedip=1"]
        V_tfilt(t), [guess=1, description="Voltage after filter"]
        V_tfiltlim(t), [guess=1, description="Voltage after filter with lower limit 0.01"]
        ΔV_t(t), [guess=0, description="Difference between filterd terminal voltage and reference voltage"]
        ΔV_tdbd(t), [guess=0, description="Voltage after deadband"]
        I_qinj(t), [guess=0, description="Limited current injection q-Phase from Voltage"]
        if PfFlag
            P_PF(t), [guess=0.015, description="Inverter active power after filter"]
        end
        Q_con(t), [guess=-0.056656797, description="Reactive Power after PfFlag"]
        if Vflag && QFlag
            Q_lim(t), [guess=-0.056656797, description="ReactivePower after limiter"]
            ΔQ(t), [guess=4.8269116e-9, description="Difference between Q_lim and Q_gen"]
            s_Q(t), [guess=4.8269116e-9, description=""]
            V_in(t), [guess=1, description="Voltage after local Q regulator"]
            V_lima(t), [guess=1, description="Limited voltage after Q regulator"]
        end
        if QFlag
            V_con(t), [guess=1, description="Voltage after Vflag"]
            V_limb(t), [guess=0.9, description="Limited voltage V_con"]
            ΔV(t), [guess=-0.1, description="Difference between V_limb and V_tfilt"]
            s_V(t), [guess=-0.1, description=""]
            I_in(t), [guess=-0.216658, description="Current after local voltage regulator"]
            I_lim(t), [guess=-0.216658, description="limited current after voltage regulator"]
        end
        I_t(t), [guess=-0.056656797, description="Current from Q_con/V_tfiltlim"]
        ΔI(t), [guess=0, description=""]
        I_qin(t), [guess=-0.056656797, description="Current after Reactive current regulator"]
        I_qcon(t), [guess=-0.056656797, description="Current after QFlag"]
        I_sum(t), [guess=-0.056656797, description="sum of I_qcon and I_qinj"]
        I_qcmd(t), [guess=-0.056656797, description="q-Phase output current"]
        P_in(t), [guess=0.015, description=""]
        P_refout(t), [guess=0.015, description="Active power after inverter power order"]
        P_lim(t), [guess=0.015, description="Limited active power after inverter power order"]
        ΔP(t), [guess=0, description="Active power difference between P_ref and P_refout"]
        ΔP_lim(t), [guess=0, description="Ramp-limited active power difference"]
        I_pref(t), [guess=0.015, description="Current from P_lim/V_tfiltlim"]
        I_pcmd(t), [guess=0.015, description="p-Phase output current"]
        I_qmin(t), [guess=-1.1, description="Minumum q-Phase current limit (pu)"]
        I_qmax(t), [guess=1.1, description="Maximum q-Phase current limit (pu)"]
        I_pmax(t), [guess=0.9259688073, description="Maximum p-Phase current limit (pu)"]
        I_pmin(t), [guess=0, description="Minumum p-Phase current limit (pu)"]
        I_pre(t), [guess=0.8574182321, description=""]
        I_post(t), [guess=0.9259688073, description=""]
        VDL1_out(t), [guess=1.1, description=""]
        VDL2_out(t), [guess=1.2, description=""]
    end
    @equations begin
        Voltage_dip ~ ifelse(Vt_in.u<V_dip, 1, ifelse(Vt_in.u>V_up, 1, 0))

        simpleLag.in ~ Vt_in.u
        V_tfilt ~ simpleLag.out

        V_tfiltlim ~ max(V_tfilt, 0.01)
        ΔV_t ~ V_ref0 - V_tfilt

        deadband.in ~ ΔV_t
        ΔV_tdbd ~ deadband.out

        I_qinj ~ (1-Voltage_dip) * clamp(K_qv*ΔV_tdbd, I_ql1, I_qh1)

        if PfFlag
            simpleLag1.in ~ P_e.u
            P_PF ~ simpleLag1.out
            Q_con ~ P_PF * tan(P_faref.u)
        else
            Q_con ~ Qext_in.u
        end

        if Vflag && QFlag
            Q_lim ~ clamp(Q_con, Q_min, Q_max)
            ΔQ ~ Q_lim - Q_gen.u
            Dt(s_Q) ~ K_qi * (1-Voltage_dip) * ΔQ
            V_in ~ K_qp * ΔQ + s_Q
            V_lima ~ clamp(V_in, V_min, V_max)
            V_con ~ V_lima
        end
        if !Vflag && QFlag
            # NOTE (2026-08-13): OpenIPSL's REECA1.mo does NOT feed the raw V_bias
            # parameter into V_con for this flag combination. Instead it computes
            # (REECA1.mo, ~line 184):
            #   Vmod = if pfflag==false and vflag==false and qflag==true
            #            then V0 - PfFlag.y   # PfFlag.y ≡ Qext ≡ our Q_con
            #            else Vbias
            #   V_con = Q_con + Vmod
            # Plugging Vmod into V_con for the !PfFlag branch gives
            # V_con = Q_con + (V_0 - Q_con) ≡ V_0 -- Q_con cancels analytically, so
            # we write the reduced form directly.
            #
            # Previously this branch read `V_con ~ Q_con + V_bias` with V_bias fixed
            # at 0, i.e. it assigned a reactive-power-scaled value (~-0.05 pu)
            # directly to a voltage-scaled target (expected ~1 pu), which structurally
            # clamped V_limb at V_min. That was the real, model-level root cause of
            # WT4B's QFlag=true initialization failure (confirmed by cross-checking
            # against a validated OpenModelica reference run: with this fix, WT4B's
            # reeca.s_V converges to -0.056657, matching the reference's -0.056658 to
            # 5 decimals, with zero overrides needed). WT4B still needs
            # `subalg=LevenbergMarquardt()` to initialize at all -- that is a separate,
            # pre-existing default-solver robustness issue affecting WT4B regardless
            # of QFlag (confirmed: baseline QFlag=false also fails with the default
            # solver), not something this fix changes or depends on.
            if !PfFlag
                V_con ~ V_0
            else
                V_con ~ Q_con + V_bias
            end
        end
        if QFlag
            V_limb ~ clamp(V_con, V_min, V_max)
            ΔV ~ V_limb - V_tfilt
            Dt(s_V) ~ (1-Voltage_dip) * K_vi * ΔV
            I_in ~ K_vp * ΔV + s_V
            I_lim ~ clamp(I_in, I_qmin, I_qmax)
            I_qcon ~ I_lim
        else
            I_qcon ~ I_qin
        end

        I_t ~ Q_con / V_tfiltlim
        ΔI ~ I_t - I_qin
        T_iq * Dt(I_qin) ~ ΔI
        I_sum ~ I_qcon + I_qinj
        I_qcmd ~ clamp(I_sum, I_qmin, I_qmax)
        #p-phase current
        # NOTE (2026-08-17): keyed on PFlag (OpenIPSL `pflag`), NOT on PfFlag
        # (`pfflag`) -- see the comment on the structural parameters above.
        if PFlag
            P_in ~ Wg.u * Pref_in.u
        else
            P_in ~ Pref_in.u
        end
        ΔP ~ P_in - P_refout
        ΔP_lim ~ clamp(ΔP, dP_min, dP_max)
        T_pord * Dt(P_refout) ~ ΔP_lim
        P_lim ~ clamp(P_refout, P_min, P_max)
        I_pref ~ P_lim/V_tfiltlim
        I_pcmd ~ clamp(I_pref, I_pmin, I_pmax)
        #VDL tables
        VDL1_out ~ VDL(V_tfilt, Vq1, Vq2, Vq3, Vq4, Iq1, Iq2, Iq3, Iq4)
        VDL2_out ~ VDL(V_tfilt, Vp1, Vp2, Vp3, Vp4, Ip1, Ip2, Ip3, Ip4)
        #current limiter logic
        I_pmin ~ 0
        I_qmin ~ -I_qmax
        I_pre ~ ifelse(PqFlag, (sqrt(I_max) - sqrt(abs(I_pcmd))), (sqrt(I_max) - sqrt(abs(I_qcmd))))
        I_post ~ ifelse(I_pre<0, 0, sqrt(I_pre))
        I_pmax ~ ifelse(PqFlag, min(VDL2_out, I_max), min(VDL2_out, I_post))
        I_qmax ~ ifelse(PqFlag, min(VDL1_out, I_post), min(VDL1_out, I_max))
        #outputs
        Iqcmd_out.u ~ I_qcmd
        Ipcmd_out.u ~ I_pcmd
    end
end


@mtkmodel reec_b_pf begin
    @structural_parameters begin
        PfFlag = false
        Vflag = false
        QFlag = false
        PqFlag = false
    end
    @parameters begin
        V_dip, [description="Low voltage condition trigger voltage (pu)"]
        V_up, [description="High voltage condition trigger voltage (pu)"]
        T_rv, [description="Terminal bus voltage filter time constant (s)"]
        V_ref0, [description="Reference voltage for reactive current injection (pu)"]
        dbd1, [description="Overvoltage deadband for reactive current injection (pu)"]
        dbd2, [description="Undervoltage deadband for reactive current injection (pu)"]
        K_qv, [description="Reactive current injection gain (pu/pu)"]
        I_qh1, [description="Maximum reactive current injection (pu on mbase)"]
        I_ql1, [description="Minimum reactive current injection (pu on mbase)"]
        T_p, [description="Active power filter time constant (s)"]
        Q_min, [description="Minimum reactive power when Vflag = 1 (pu on mbase)"]
        Q_max, [description="Maximum reactive power when Vflag = 1 (pu on mbase)"]
        V_min, [description="Minimum voltage at inverter terminal bus (pu)"]
        V_max, [description="Maximum voltage at inverter terminal bus (pu)"]
        K_qp, [description="Local Q regulator proportional gain (pu/pu)"]
        K_qi, [description="Local Q regulator integral gain (pu/pu-s)"]
        K_vp, [description="Local voltage regulator proportional gain (pu/pu)"]
        K_vi, [description="Local voltage regulator integral gain (pu/pu-s)"]
        I_max, [description="Maximum apparent current (pu on mbase)"]
        T_iq, [description="Reactive current regulator lag time constant (s)"]
        T_pord, [description="Inverter power order lag time constant (s)"]
        P_min, [description="Minimum active power (pu on mbase)"]
        P_max, [description="Maximum active power (pu on mbase)"]
        dP_min, [description="Active power down-ramp limit (pu/s on mbase)"]
        dP_max, [description="Active power up-ramp limit (pu/s on mbase)"]
    end
    @components begin
        Vt_in = RealInput(guess=1.001047)
        P_e = RealInput(guess=0.800721) #Inverter active power (pu on mbase)
        P_initial = RealInput(guess=0.800721)
        Q_initial = RealInput(guess=-0.30027)
        Qext_in = RealInput(guess=-0.30027)
        Pref_in = RealInput(guess=0.800721)
        Q_gen = RealInput(guess=-0.30027)
        # outputs
        Iqcmd_out = RealOutput(guess=-0.299956)
        Ipcmd_out = RealOutput(guess=0.799884)
        Vt_filt = RealOutput(guess=1.001047)

        #building blocks
        simpleLag = PowerDynamics.Library.SimpleLag(K=1, T=T_rv, guess=1.001047)
        simpleLag1 = PowerDynamics.Library.SimpleLag(K=1, T=T_p, guess=0.800721)
        deadband = PowerDynamics.Library.DeadZone(uMax=dbd2, uMin=dbd1)
        if Vflag && QFlag
            PI_freeze = PowerDynamics.Library.P_I_Lim_freeze(K_p=K_qp, K_i=K_qi, T=1, outMin=V_min, outMax=V_max, guess=1.001047, guessx=1.001047)
        end
        if QFlag
            PI_freeze_var = PowerDynamics.Library.P_I_varLim_freeze(K_p=K_vp, K_i=K_vi, T=1, guess=-0.299956, guessx=-0.299956)
        end
        simpleLag_freeze = PowerDynamics.Library.SimpleLag_freeze(K=1, T=T_iq, guess=-0.299956, guessx=-0.299956)
        P_limLag = PowerDynamics.Library.SimpleLag_2Lims_freeze(K=1, T=T_pord, doutMin=dP_min, doutMax=dP_max, outMin=P_min, outMax=P_max, guess=0.800721, guessin=0.800721, guessx=0.800721)
    end
    @variables begin
        Voltage_dip(t), [guess=0, description="freeze states if Voltagedip=1"]
        V_tfilt(t), [guess=1.001047, description="Voltage after filter"]
        V_tfiltlim(t), [guess=1.001047, description="Voltage after filter with lower limit 0.01"]
        ΔV_t(t), [guess=1.001047, description="Difference between filterd terminal voltage and reference voltage"]
        ΔV_tdbd(t), [guess=-0.981046, description="Voltage after deadband"]
        I_qinj(t), [guess=0, description="Limited current injection q-Phase from Voltage"]
        P_PF(t), [guess=0.800721, description="Inverter active power after filter"]
        Q_con(t), [guess=-0.30027, description="Reactive Power after PfFlag"]
        if Vflag && QFlag
            Q_lim(t), [guess=-0.30027, description="ReactivePower after limiter"]
            ΔQ(t), [guess=0, description="Difference between Q_lim and Q_gen"]
            V_lima(t), [guess=1.001047, description="Limited voltage after Q regulator"]
        end
        if QFlag
            V_con(t), [guess=1.001047, description="Voltage after Vflag"]
            V_limb(t), [guess=1.001047, description="Limited voltage V_con"]
            ΔV(t), [guess=0, description="Difference between V_limb and V_tfilt"]
            I_lim(t), [guess=-0.299956, description="limited current after voltage regulator"]
        end
        I_t(t), [guess=-0.299956, description="Current from Q_con/V_tfiltlim"]
        #ΔI(t), [guess=-1.9310942e-14, description=""]
        I_qin(t), [guess=-0.299956, description="Current after Reactive current regulator"]
        I_qcon(t), [guess=-0.299956, description="Current after QFlag"]
        I_sum(t), [guess=-0.299956, description="sum of I_qcon and I_qinj"]
        I_qcmd(t), [guess=-0.299956, description="q-Phase output current"]
        #P_refout(t), [guess=0.015, description="Active power after inverter power order"]
        P_lim(t), [guess=0.800721, description="Limited active power after inverter power order"]
        #ΔP(t), [guess=-7.3598558e-12, description="Active power difference between P_ref and P_refout"]
        #ΔP_lim(t), [guess=-7.3598558e-12, description="Ramp-limited active power difference"]
        I_pref(t), [guess=0.799884, description="Current from P_lim/V_tfiltlim"]
        I_pcmd(t), [guess=0.799884, description="p-Phase output current"]
        I_qmin(t), [guess=-1.25, description="Minumum q-Phase current limit (pu)"]
        I_qmax(t), [guess=1.25, description="Maximum q-Phase current limit (pu)"]
        I_pmax(t), [guess=1.213477, description="Maximum p-Phase current limit (pu)"]
        I_pmin(t), [guess=0, description="Minumum p-Phase current limit (pu)"]
    end
    @equations begin
        #T_rv * Dt(V_tfilt) ~ Vt_in.u - V_tfilt
        simpleLag.in ~ Vt_in.u
        V_tfilt ~ simpleLag.out
        Voltage_dip ~ ifelse(V_tfilt<V_dip, 1, ifelse(V_tfilt>V_up, 1, 0)) #TODO in PF wird Voltage_dip mit V_tfilt bestimmt

        V_tfiltlim ~ max(V_tfilt, 0.01) #lowlimit(V_tfilt, 0.01)
        #q-phase current
        ΔV_t ~ - V_ref0 + V_tfilt #TODO in PF VZ anders im Vgl zur angegebenen Quelle

        #ΔV_tdbd ~ deadband(ΔV_t, dbd1, dbd2)
        deadband.in ~ ΔV_t
        ΔV_tdbd ~ deadband.out

        I_qinj ~ clamp(K_qv*ΔV_tdbd, I_ql1, I_qh1) #limiter(K_qv*ΔV_tdbd, I_ql1, I_qh1)

        #T_p * Dt(P_PF) ~ P_e.u - P_PF
        simpleLag1.in ~ P_e.u
        P_PF ~ simpleLag1.out

        Q_con ~ ifelse(PfFlag, P_PF * Q_initial.u/P_initial.u, Qext_in.u)  #ifelse(PfFlag, P_PF * tan(P_faref.u), Qext_in.u)
        if Vflag && QFlag
            Q_lim ~ clamp(Q_con, Q_min, Q_max)
            ΔQ ~ Q_lim - Q_gen.u
            PI_freeze.in ~ ΔQ
            PI_freeze.freeze ~ Voltage_dip
            V_lima ~ PI_freeze.pi_out
            V_con ~ V_lima
        end
        if !Vflag && QFlag
            # NOTE (2026-08-13, OPEN ISSUE): `V_con ~ Q_con` was checked against
            # PowerFactory's REEC_B implementation and confirmed CORRECT -- this is
            # NOT the same bug as reec_a's missing Vmod (do not "fix" this again
            # without new evidence). Despite the equation being right, the flag
            # combination Vflag=false + QFlag=true on WECC_large_PV_pf initializes
            # with a much worse residual (~0.11) than typical clean solves (~1e-5),
            # even though it still passes the network's actual tol=1e0 threshold.
            # Investigated and RULED OUT as causes: (1) K_vp=0 degeneracy (tested
            # K_vp=1e-10/1e-6/1e-2 -- residual identical to 14 digits, unlike the
            # analogous reec_c/BESS case where this was the actual fix); (2)
            # saturation/clamping at this specific operating point (tested Q from
            # -0.3333 up to 1.0, i.e. inside/outside the V_min/V_max clamp band --
            # residual got WORSE (up to ~8.7), not better, moving away from the
            # original point); (3) obvious structural degeneracy (mtkcompile
            # comparison against the working Vflag=true,QFlag=false baseline shows
            # only the expected extra PI_freeze_var state, nothing resembling the
            # reec_c K_vp*s_V pattern). Root cause still unknown -- deprioritized,
            # not further investigated per user decision. PowerFactory apparently
            # initializes this combination cleanly, so this is likely solvable;
            # revisit with a per-equation residual breakdown if this becomes
            # relevant again.
            V_con ~ Q_con
        end
        if QFlag
            V_limb ~ clamp(V_con, V_min, V_max)
            ΔV ~ V_limb - V_tfilt
            PI_freeze_var.in ~ ΔV
            PI_freeze_var.freeze ~ Voltage_dip
            PI_freeze_var.min ~ I_qmin
            PI_freeze_var.max ~ I_qmax
            I_lim ~ PI_freeze_var.pi_out
            I_qcon ~ I_lim
        else
            I_qcon ~ I_qin
        end
        I_t ~ Q_con / V_tfiltlim
        simpleLag_freeze.in ~ I_t
        simpleLag_freeze.freeze ~ Voltage_dip
        I_qin ~ simpleLag_freeze.out

        I_sum ~ I_qcon + I_qinj
        I_qcmd ~ limiter(I_sum, I_qmin, I_qmax) #clamp(I_sum, I_qmin, I_qmax) #limiter(I_sum, I_qmin, I_qmax)

        #p-phase current
        P_limLag.in ~ Pref_in.u
        P_limLag.freeze ~ Voltage_dip
        P_lim ~ P_limLag.out
        I_pref ~ P_lim/V_tfiltlim
        I_pcmd ~ limiter(I_pref, I_pmin, I_pmax) #clamp(I_pref, I_pmin, I_pmax) #limiter(I_pref, I_pmin, I_pmax)

        #current limiter logic
        I_pmin ~ 0
        I_qmin ~ - I_qmax
        I_pmax ~ ifelse(PqFlag, I_max, sqrt(I_max^2 - min(I_sum^2, I_max^2))) #ifelse(PqFlag, I_max, sqrt(I_max^2 - min(I_qcmd^2, I_max^2)))
        I_qmax ~ ifelse(PqFlag, sqrt(I_max^2 - min(I_pref^2, I_max^2)), I_max) #ifelse(PqFlag, sqrt(I_max^2 - min(I_pcmd^2, I_max^2)), I_max)
        #outputs
        Iqcmd_out.u ~ I_qcmd
        Ipcmd_out.u ~ I_pcmd
        Vt_filt.u ~ V_tfilt
    end
end
