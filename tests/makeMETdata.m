function makeMETdata(source_path, output_path)
% testData(source_path, audio_file, output_path)
%   source_path = path to METLab code
%   output_path = directory to store generated files
%
% CREATED:2015-02-16 12:50:31 by Brian McFee <brian.mcfee@nyu.edu>
%  Generate the test suite data for spectral features used by
%  the MET lab: http://music.ece.drexel.edu/
%
%   spectral_centroid
%   spectral_contrast
%   spectral_bandwidth
%   spectral_rolloff
%   
%
% NOTE:
%   This script is separate from makeTestData.m because it's designed for use in
%   octave, rather than matlab.

    addpath(source_path)

    display('spectral_centroid')
    testCentroid(output_path);

    display('spectral_bandwidth')
    testBandwidth(output_path);
    
    display('spectral_contrast')
    testContrast(output_path);
    
    display('spectral_rolloff')
    testRolloff(output_path);

    display('Done.')
end


function testCentroid(output_path)

    wavfile = 'data/test1_22050.wav';

    [y_in, sr]  = wavread(wavfile);
    y_in = mean(y_in, 2);

    P_NFFT  = [512];
    P_HOP   = [256];

    counter = 0;
    for nfft = P_NFFT
        for hop_length = P_HOP
            centroid = spectralCentroid(y_in, sr, nfft, hop_length, nfft);

            counter = counter + 1;

            filename = sprintf('%s/met-centroid-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename])
            save('-7', filename, 'centroid', 'nfft', 'hop_length', 'wavfile');
        end
    end
end


function testContrast(output_path)

    wavfile = 'data/test1_22050.wav';

    [y_in, sr]  = wavread(wavfile);
    y_in = mean(y_in, 2);

    P_NFFT  = [512];
    P_HOP   = [256];

    counter = 0;
    for nfft = P_NFFT
        for hop_length = P_HOP
            contrast = spectralContrast(y_in, sr, nfft, hop_length, nfft);

            counter = counter + 1;

            filename = sprintf('%s/met-contrast-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename])
            save('-7', filename, 'contrast', 'nfft', 'hop_length', 'wavfile');
        end
    end
end


function testRolloff(output_path)

    wavfile = 'data/test1_22050.wav';

    [y_in, sr]  = wavread(wavfile);
    y_in = mean(y_in, 2);

    P_NFFT  = [512];
    P_HOP   = [256];
    P_PCT   = [0.8, 0.9, 0.95];

    counter = 0;
    for nfft = P_NFFT
        for hop_length = P_HOP
            for pct = P_PCT
                rolloff = spectralRolloff(y_in, sr, nfft, hop_length, nfft, pct);

                counter = counter + 1;
    
                filename = sprintf('%s/met-rolloff-%03d.mat', output_path, counter);
                display(['  `-- saving ', filename])
                save('-7', filename, 'rolloff', 'nfft', 'hop_length', 'pct', 'wavfile');
            end
        end
    end
end


function testBandwidth(output_path)

    wavfile = 'data/test1_22050.wav';

    [y_in, sr]  = wavread(wavfile);
    y_in = mean(y_in, 2);

    P_NFFT  = [512];
    P_HOP   = [256];

    counter = 0;
    for nfft = P_NFFT
        for hop_length = P_HOP
            S = spectrogram(y_in, sr, nfft, hop_length, nfft);
            centroid = spectralCentroid(y_in, sr, nfft, hop_length, nfft);
            bw = spectralBandwidth(S, centroid, sr);

            counter = counter + 1;

            filename = sprintf('%s/met-bandwidth-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename])
            save('-7', filename, 'bw', 'S', 'centroid', 'nfft', 'hop_length', 'wavfile');
        end
    end
end
