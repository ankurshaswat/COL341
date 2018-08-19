#!/bin/bash
# 1. Put your files in submissions directory and run the script as follows:
# ./evaluate_q2.sh <path_to_data_dir> <path_to_sandbox_dir> <entry_number> <path_to_submissions_dir>
# Example:
# ./evaluate_q2.sh $HOME/data $HOME/sandbox 2016ANZ8048 $HOME/submissions

run()
{
	: '
        Args:        
	    	$1 file name
	    	$2 part
	    	$3 method
	    	$4 learning rate
	    	$5 num_iterations
	    	$6 batch_size
	    	$7 train data
	    	$8 vocab_file
	    	$9 test data
	    	$10 output file
    '
    chmod +x $1
    ./$1 $2 $3 $4 $5 $6 $7 $8 $9 "${10}"
}


compute_score()
{
    : '
        Compute score as per predicted values and write to given file
        $1 python_file
        $2 targets
        $3 predicted
        $4 outfile
    '
    python3 $1 $2 $3 $4
}

main()
{
    : '
        $1: data_dir
        $2: sandbox_dir
        $3: entry_number
        $4: submissions_dir
    '
    main_dir=`pwd`
    unzip $4/$3.zip -d $2
    lr=0.05
    num_iter=500
    batch_size=128    
    # Run Q2
    cd $2/$3
    for part in a b; do
        for algo in 1 2 3; do
            run logreg $part $algo $lr $num_iter $batch_size $1/imdb_train.csv $1/imdb_vocab $1/imdb_test.csv $2/$3/results_${part}_${algo}
            compute_score $main_dir/compute_accuracy.py $1/imdb_test.csv $2/$3/results_${part}_${algo}
        done
    done
    cd $main_dir

}

main $1 $2 $3 $4
