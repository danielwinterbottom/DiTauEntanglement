#!/bin/sh
echo "Cluster = $1 Process = $2"
cd /vols/cms/dw515/HH_reweighting/DiTauEntanglement/

if [ $2 == 0 ]; then

	model="LEP_mlp_inc_reco_taus_onorm_Jan08_newLRschedule_v2"
	python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --useMLP --output_name output_inc_entanglement &> outputs_${model}/out_inc_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

elif [ $2 == 1 ]; then

        model="LEP_mlp_inc_reco_taus_Jan08_newLRschedule_v2"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --useMLP --output_name output_inc_entanglement &> outputs_${model}/out_inc_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

elif [ $2 == 2 ]; then

        model="LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --output_name output_inc_entanglement &> outputs_${model}/out_inc_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 3 ]; then

        model="LEP_nflow_inc_reco_taus_Jan08_newLRschedule"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --output_name output_inc_entanglement &> outputs_${model}/out_inc_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 4 ]; then

        model="LEP_mlp_inc_reco_taus_onorm_Jan08_newLRschedule_v2"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_onorm_dataframe_reduced.pkl --useMLP --output_name output_no_entanglement &> outputs_${model}/out_no_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

elif [ $2 == 5 ]; then

        model="LEP_mlp_inc_reco_taus_Jan08_newLRschedule_v2"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_dataframe_reduced.pkl --useMLP --output_name output_no_entanglement &> outputs_${model}/out_no_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

elif [ $2 == 6 ]; then

        model="LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_onorm_dataframe_reduced.pkl --output_name output_no_entanglement &> outputs_${model}/out_no_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 7 ]; then

        model="LEP_nflow_inc_reco_taus_Jan08_newLRschedule"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_dataframe_reduced.pkl --output_name output_no_entanglement &> outputs_${model}/out_no_entanglement.out

        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 8 ]; then

        model="LEP_mlp_inc_reco_taus_onorm_Jan08_newLRschedule_v2"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_onorm_dataframe_reduced.pkl --useMLP --output_name output_uncorrelated &> outputs_${model}/out_uncorrelated.out

        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

elif [ $2 == 9 ]; then

        model="LEP_mlp_inc_reco_taus_Jan08_newLRschedule_v2"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_reduced_dataframe.pkl --useMLP --output_name output_uncorrelated &> outputs_${model}/out_uncorrelated.out

        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

elif [ $2 == 10 ]; then

        model="LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_onorm_dataframe_reduced.pkl --output_name output_uncorrelated &> outputs_${model}/out_uncorrelated.out

        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_for_pred_${chan}.out
        done

elif [ $2 == 11 ]; then

        model="LEP_nflow_inc_reco_taus_Jan08_newLRschedule"
        python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_reduced_dataframe.pkl --output_name output_uncorrelated &> outputs_${model}/out_uncorrelated.out

        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_for_pred_${chan}.out
        done

fi
