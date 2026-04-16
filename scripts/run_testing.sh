#!/bin/sh
echo "Cluster = $1 Process = $2"
cd /vols/cms/dw515/HH_reweighting/DiTauEntanglement/

#model_nflow="LEP_nflow_inc_reco_taus_Jan08_newLRschedule"
model_nflow="LEP_nflow_inc_reco_taus_Dec19_cont1"
model_nflow_onorm="LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule"
model_mlp="LEP_mlp_inc_reco_taus_Jan08_newLRschedule_v2"
model_mlp_onorm="LEP_mlp_inc_reco_taus_onorm_Jan08_newLRschedule_v2"

if [ $2 == 100 ]; then
        model=$model_nflow

        # these lines store the true and analytical values
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix true &> outputs_${model}/out_inc_entanglement_true.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix true --channel ${chan} &> outputs_${model}/out_inc_entanglement_true_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix ana_pred &> outputs_${model}/out_inc_entanglement_ana_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix ana_pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_ana_pred_${chan}.out
        done

        # do the same for no entanglement case
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix true &> outputs_${model}/out_no_entanglement_true.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix true --channel ${chan} &> outputs_${model}/out_no_entanglement_true_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix ana_pred &> outputs_${model}/out_no_entanglement_ana_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix ana_pred --channel ${chan} &> outputs_${model}/out_no_entanglement_ana_pred_${chan}.out
        done

        # now do the same for the uncorrelated case
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix true &> outputs_${model}/out_uncorrelated_true.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix true --channel ${chan} &> outputs_${model}/out_uncorrelated_true_${chan}.out
        done

        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix ana_pred &> outputs_${model}/out_uncorrelated_ana_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix ana_pred --channel ${chan} &> outputs_${model}/out_uncorrelated_ana_pred_${chan}.out
        done

fi

if [ $2 == 200 ]; then
        # run true and analytical for gamma and higgs samples
        model=$model_nflow
        for prefix in "gamma_inc_entanglement" "higgs_inc_entanglement"; do
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_${prefix}.root --prefix true &> outputs_${model}/out_${prefix}_true.out
            for chan in 'pipi' 'rhorho' 'pirho'; do
                    python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_${prefix}.root --prefix true --channel ${chan} &> outputs_${model}/out_${prefix}_true_${chan}.out
            done

            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_${prefix}.root --prefix ana_pred &> outputs_${model}/out_${prefix}_ana_pred.out
            for chan in 'pipi' 'rhorho' 'pirho'; do
                    python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_${prefix}.root --prefix ana_pred --channel ${chan} &> outputs_${model}/out_${prefix}_ana_pred_${chan}.out
            done
        done
fi

if [ $2 == 0 ]; then

	model=$model_mlp_onorm
        echo "Using model: ${model}"
        echo "Running inference with entanglement included..."
	#python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --useMLP --output_name output_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

elif [ $2 == 1 ]; then

        model=$model_mlp
        echo "Using model: ${model}"
        echo "Running inference with entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --useMLP --output_name output_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

elif [ $2 == 2 ]; then

        model=$model_nflow_onorm
        echo "Using model: ${model}"
        echo "Running inference with entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --output_name output_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan} (pred)"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 3 ]; then

        model=$model_nflow
        echo "Using model: ${model}"
        echo "Running inference with entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name outputs_LEP_nflow_inc_reco_taus_onorm_Jan08_newLRschedule_cont1/test_dataframe.pkl --output_name output_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred &> outputs_${model}/out_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 4 ]; then

        model=$model_mlp_onorm
        echo "Using model: ${model}"
        echo "Running inference without entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_onorm_dataframe_reduced.pkl --useMLP --output_name output_no_entanglement &> outputs_${model}/LEP_NF_reco_out_no_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

elif [ $2 == 5 ]; then

        model=$model_mlp
        echo "Using model: ${model}"
        echo "Running inference without entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_dataframe_reduced.pkl --useMLP --output_name output_no_entanglement &> outputs_${model}/LEP_NF_reco_out_no_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

elif [ $2 == 6 ]; then

        model=$model_nflow_onorm

        echo "Using model: ${model}"
        echo "Running inference without entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_onorm_dataframe_reduced.pkl --output_name output_no_entanglement &> outputs_${model}/LEP_NF_reco_out_no_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 7 ]; then

        model=$model_nflow
        echo "Using model: ${model}"
        echo "Running inference without entanglement included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_no_entanglement_30M_dataframe_reduced.pkl --output_name output_no_entanglement &> outputs_${model}/LEP_NF_reco_out_no_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_no_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred &> outputs_${model}/out_no_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_no_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred &> outputs_${model}/out_no_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_no_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_no_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 8 ]; then

        model=$model_mlp_onorm
        echo "Using model: ${model}"
        echo "Running inference without spin correlations included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_onorm_dataframe_reduced.pkl --useMLP --output_name output_uncorrelated &> outputs_${model}/LEP_NF_reco_out_uncorrelated.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

elif [ $2 == 9 ]; then

        model=$model_mlp
        echo "Using model: ${model}"
        echo "Running inference without spin correlations included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_reduced_dataframe.pkl --useMLP --output_name output_uncorrelated &> outputs_${model}/LEP_NF_reco_out_uncorrelated.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

elif [ $2 == 10 ]; then

        model=$model_nflow_onorm
        echo "Using model: ${model}"
        echo "Running inference without spin correlations included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_onorm_dataframe_reduced.pkl --output_name output_uncorrelated &> outputs_${model}/LEP_NF_reco_out_uncorrelated.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_for_pred_${chan}.out
        done

elif [ $2 == 11 ]; then

        model=$model_nflow
        echo "Using model: ${model}"
        echo "Running inference without spin correlations included..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_tauhtauh_uncorrelated_30M_reduced_dataframe.pkl --output_name output_uncorrelated &> outputs_${model}/LEP_NF_reco_out_uncorrelated.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_uncorrelated.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred &> outputs_${model}/out_uncorrelated.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_uncorrelated_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred &> outputs_${model}/out_uncorrelated_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_uncorrelated.root --prefix pred --channel ${chan} &> outputs_${model}/out_uncorrelated_for_pred_${chan}.out
        done

elif [ $2 == 12 ]; then

        model=$model_mlp_onorm
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (gamma only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_gamma_to_tauhtauh_10M_onorm_dataframe_reduced.pkl --useMLP --output_name output_gamma_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_gamma_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_gamma_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred &> outputs_${model}/out_gamma_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_gamma_inc_entanglement_${chan}.out
        done
        
elif [ $2 == 13 ]; then 

        model=$model_mlp
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (gamma only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_gamma_to_tauhtauh_10M_dataframe_reduced.pkl --useMLP --output_name output_gamma_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_gamma_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_gamma_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred &> outputs_${model}/out_gamma_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_gamma_inc_entanglement_${chan}.out
        done

elif [ $2 == 14 ]; then

        model=$model_nflow_onorm
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (gamma only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_gamma_to_tauhtauh_10M_onorm_dataframe_reduced.pkl --output_name output_gamma_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_gamma_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_gamma_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_gamma_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_gamma_inc_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred &> outputs_${model}/out_gamma_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_gamma_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 15 ]; then

        model=$model_nflow
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (gamma only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_gamma_to_tauhtauh_10M_dataframe_reduced.pkl --output_name output_gamma_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_gamma_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_gamma_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_gamma_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_gamma_inc_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred &> outputs_${model}/out_gamma_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_gamma_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_gamma_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 16 ]; then

        model=$model_mlp_onorm
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (higgs only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_higgs_to_tauhtauh_10M_onorm_dataframe_reduced.pkl --useMLP --output_name output_higgs_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_higgs_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_higgs_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred &> outputs_${model}/out_higgs_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_higgs_inc_entanglement_${chan}.out
        done
        
elif [ $2 == 17 ]; then 

        model=$model_mlp
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (higgs only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_higgs_to_tauhtauh_10M_dataframe_reduced.pkl --useMLP --output_name output_higgs_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_higgs_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_higgs_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred &> outputs_${model}/out_higgs_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
            echo "Computing entanglement variables for channel: ${chan}"
            python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_higgs_inc_entanglement_${chan}.out
        done

elif [ $2 == 18 ]; then

        model=$model_nflow_onorm
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (higgs only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --use_onorm  --dataframe_name ditau_nu_regression_ee_to_higgs_to_tauhtauh_10M_dataframe_reduced.pkl --output_name output_higgs_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_higgs_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_higgs_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_higgs_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_higgs_inc_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred &> outputs_${model}/out_higgs_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_higgs_inc_entanglement_for_pred_${chan}.out
        done

elif [ $2 == 19 ]; then

        model=$model_nflow
        echo "Using model: ${model}"
        echo "Running inference with entanglement included (higgs only)..."
        #python -u python/LEP_NF_reco.py -s 4 -m ${model} --inc_reco_taus --dataframe_name ditau_nu_regression_ee_to_higgs_to_tauhtauh_10M_dataframe_reduced.pkl --output_name output_higgs_inc_entanglement &> outputs_${model}/LEP_NF_reco_out_higgs_inc_entanglement.out

        echo "Making plots..."
        python scripts/make_plots.py -i outputs_${model}/output_higgs_inc_entanglement.root

        echo "Computing entanglement variables for all channels..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix alt_pred &> outputs_${model}/out_higgs_inc_entanglement.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan}"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix alt_pred --channel ${chan} &> outputs_${model}/out_higgs_inc_entanglement_${chan}.out
        done

        echo "Computing entanglement variables for all channels (pred)..."
        python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred &> outputs_${model}/out_higgs_inc_entanglement_for_pred.out
        for chan in 'pipi' 'rhorho' 'pirho'; do
                echo "Computing entanglement variables for channel: ${chan} (pred)"
                python -u scripts/compute_entanglement_variables.py -i outputs_${model}/output_higgs_inc_entanglement.root --prefix pred --channel ${chan} &> outputs_${model}/out_higgs_inc_entanglement_for_pred_${chan}.out
        done

fi
