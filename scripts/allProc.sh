(
	bash scripts/pdmpRun.sh 2>&1 # Raw DM, width data
	bash scripts/dmData.sh 2>&1 # Process raw inputs
	bash scripts/extractMeta.sh 2>&1
	bash scripts/updateParDMs.sh 2>&1
	bash scripts/profs.sh 2>&1
	bash scripts/quickBestProfiles.sh 2>&1

	python scripts/lmfitPulses4.py 2>&1
	python scripts/lmfitPulses4_periodic.py 2>&1
	python scripts/extractMeta_fake.py 2>&1 | tee sensitivityLimits.output
	python scripts/metaParser.py 2>&1 | tee metaParser.output
	#python scripts/parParser.py 2>&1 | tee parParser.output ^^^
	python scripts/appendixModelFits.py 2>&1 | tee appendixModelsFits.output
) 2>&1 | tee allProc.log
