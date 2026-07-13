function makeCTData(source_path, output_path)
% testData(source_path, output_path)
%   source_path = path to Chroma Toolbox code
%   output_path = directory to store generated files
%
%  Generate the test suite data for chroma features used by
%  the Chroma Toolbox: https://www.audiolabs-erlangen.de/resources/MIR/chromatoolbox
%
%   CENS
%   

    addpath(source_path)

    display('Pitch (CQT) Features')
    f_cqt = testCQT(output_path);

    display('CENS Features')
    testCENS_9_2(f_cqt, output_path);
    testCENS_21_5(f_cqt, output_path);
    testCENS_41_1(f_cqt, output_path);
    
    display('Done.')
end

function f_cqt = testCQT(output_path)
    f_audio = wav_to_audio('', 'data/', 'test1_44100.wav');
    f_cqt = audio_to_pitch_via_FB(f_audio);
    save(fullfile(output_path, 'features-CT-cqt.mat'), 'f_cqt');
end

function testCENS_41_1(f_cqt, output_path)
    parameter.winLenSmooth = 41;
    parameter.downsampSmooth = 1;
    f_CENS = pitch_to_CENS(f_cqt, parameter, []);
    
    save(fullfile(output_path, 'features-CT-CENS_41-1.mat'), 'f_CENS');
end

function testCENS_9_2(f_cqt, output_path)
    parameter.winLenSmooth = 9;
    parameter.downsampSmooth = 2;
    f_CENS = pitch_to_CENS(f_cqt, parameter, []);
    
    save(fullfile(output_path, 'features-CT-CENS_9-2.mat'), 'f_CENS');
end

function testCENS_21_5(f_cqt, output_path)
    parameter.winLenSmooth = 21;
    parameter.downsampSmooth = 5;
    f_CENS = pitch_to_CENS(f_cqt, parameter, []);
    
    save(fullfile(output_path, 'features-CT-CENS_21-5.mat'), 'f_CENS');
end