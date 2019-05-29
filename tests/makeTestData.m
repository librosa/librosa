function testData(source_path, output_path)
% testData(source_path, audio_file, output_path)
%   source_path = path to DPWE code
%   output_path = directory to store generated files
%
% CREATED:2013-03-08 14:32:21 by Brian McFee <brm2132@columbia.edu>
%   Generate the test suite data for librosa routines:
%
%       hz_to_mel
%       mel_to_hz
%       hz_to_octs
%
%       stft
%       istft
%
%       ifgram
%
%       load
%       resample
%
%       melfb
%       dctfb
%
%       localmax
%

    % Make sure we have the path to DPWE code
    addpath(source_path);

    display('hz_to_mel');
    testHz2Mel(output_path);

    display('mel_to_hz');
    testMel2Hz(output_path);

    display('hz_to_octs');
    testHzToOcts(output_path);

    display('load');
    testLoad(output_path);

    display('stft');
    testSTFT(output_path);

    display('istft');
    testISTFT(output_path);


    display('ifgram');
    testIFGRAM(output_path);

    display('melfb');
    testMelfb(output_path);

    display('chromafb');
    testChromafb(output_path);

    display('resample');
    testResample(output_path);

    display('beat');
    testBeat(output_path);

    display('lpcburg');
    testLPCBurg(output_path);

    %% Done!
    display('Done.');
end

function testHz2Mel(output_path)

    % Test with either a scalar argument or a vector
    P_HZ = {[440], [2.^(1:13)]};

    % Slaney-style or HTK
    P_HTK         = {0, 1};

    counter     = 0;
    for i = 1:length(P_HZ)
        f = P_HZ{i};

        for j = 1:length(P_HTK)
            htk = P_HTK{j};
        
            % Run the function
            result = hz2mel(f, htk);

            % save the output
            counter = counter + 1;

            filename = sprintf('%s/feature-hz_to_mel-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);

            save(filename, 'f', 'htk', 'result');
        end
    end
end

function testMel2Hz(output_path)

    % Test with either a scalar argument or a vector
    P_MELS      = {[5], [2.^(-2:9)]};

    % Slaney-style or HTK
    P_HTK       = {0, 1};

    counter     = 0;
    for i = 1:length(P_MELS)
        f = P_MELS{i};

        for j = 1:length(P_HTK)
            htk = P_HTK{j};
        
            % Run the function
            result = mel2hz(f, htk);

            % save the output
            counter = counter + 1;

            filename = sprintf('%s/feature-mel_to_hz-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);

            save(filename, 'f', 'htk', 'result');
        end
    end
end

function testHzToOcts(output_path)

    % Scalar argument or a vector
    P_HZ      = {[5], [2.^(2:14)]};

    counter     = 0;
    for i = 1:length(P_HZ)
        f = P_HZ{i};

        % Run the function
        result = hz2octs(f);

        % save the output
        counter = counter + 1;

        filename = sprintf('%s/feature-hz_to_octs-%03d.mat', output_path, counter);
        display(['  `-- saving ', filename]);

        save(filename, 'f', 'result');
    end
end

function testLoad(output_path)

    % Test: load a wav file
    %       get audio stream (floats) and sample rate
    %       preserve stereo or convert to mono
    wavfile         = 'data/test1_44100.wav';
    [y, sr]         = wavread(wavfile);
    y = y';         % Transpose to make python code easier
    mono            = 0;

    % Stereo output
    counter = 1;

    filename = sprintf('%s/core-load-%03d.mat', output_path, counter);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 'mono', 'y', 'sr');

    % Mono output
    counter = 2;
    mono    = 1;
    y       = mean(y, 1);
    filename = sprintf('%s/core-load-%03d.mat', output_path, counter);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 'mono', 'y', 'sr');

end

function testMelfb(output_path)

    % Test three sample rates
    P_SR        = [8000, 11025, 22050];

    % Two FFT lengths
    P_NFFT      = [256, 512];

    % Three filter bank sizes
    P_NFILTS    = [20, 40, 120];

    % One width
    P_WIDTH     = [1.0];

    % F_min
    P_FMIN      = [0, 512];

    % F_max
    P_FMAX      = [2000, inf];

    % Slaney or HTK mels
    P_HTK       = [0, 1];

    % Generate tests
    counter = 0;

    for sr = P_SR
        for nfft = P_NFFT
            for nfilts = P_NFILTS
                for width = P_WIDTH
                    for fmin = P_FMIN
                        for fmax = P_FMAX
                            if isinf(fmax)
                                fmax = sr / 2;
                            end
                            for htk = P_HTK

            % Run the function
            [wts, frqs] = fft2melmx(nfft, sr, nfilts, width, fmin, fmax, htk, 0);

            % save the output
            counter = counter + 1;

            filename = sprintf('%s/feature-melfb-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);

            save(filename, ...
                    'sr', 'nfft', 'nfilts', 'width', ...
                    'fmin', 'fmax', 'htk', 'wts', 'frqs');
                            end
                        end
                    end
                end
            end
        end
    end

end

function testResample(output_path)

    wavfile         = 'data/test1_22050.wav';

    [y_in, sr_in]   = wavread(wavfile);
    y_in            = mean(y_in, 2);        % Convert to mono

    % Test a downsample, same SR, and upsample
    P_SR            = [8000, 22050, 44100];

    counter = 0;

    for sr_out = P_SR

        y_out       = resample(y_in, sr_out, sr_in);

        counter     = counter + 1;
        filename    = sprintf('%s/core-resample-%03d.mat', output_path, counter);
        display(['  `-- saving ', filename]);
        save(filename, 'wavfile', 'y_in', 'sr_in', 'y_out', 'sr_out');
    end
end

function testSTFT(output_path)

    wavfile     = 'data/test1_22050.wav';

    [y, sr]     = wavread(wavfile);
    y           = mean(y, 2);        % Convert to mono

    % Test a couple of different FFT window sizes
    P_NFFT      = [128, 256, 1024];

    % And hop sizes
    P_HOP       = [64, 128, 256];

    % Note: librosa.stft does not support user-supplied windows,
    %       so we do not generate tests for this case.

    counter     = 0;

    for nfft = P_NFFT
        for hop_length = P_HOP
            % Test once with no hann window (rectangular)
            hann_w = 0;
            D = stft(y, nfft, hann_w, hop_length, sr);

            counter     = counter + 1;
            filename    = sprintf('%s/core-stft-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);
            save(filename, 'wavfile', 'D', 'sr', 'nfft', 'hann_w', 'hop_length');

            % And again with default hann window (nfft)
            hann_w      = nfft;
            D = stft(y, nfft, hann_w, hop_length, sr);

            counter     = counter + 1;
            filename    = sprintf('%s/core-stft-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);
            save(filename, 'wavfile', 'D', 'sr', 'nfft', 'hann_w', 'hop_length');
        end
    end
end

function testIFGRAM(output_path)

    wavfile     = 'data/test1_22050.wav';

    [y, sr]     = wavread(wavfile);
    y           = mean(y, 2);        % Convert to mono

    % Test a couple of different FFT window sizes
    P_NFFT      = [1024];

    % And window sizes
%     P_WIN       = [0.25, 0.5, 1.0];
    P_WIN       = [1.0];

    % And hop sizes
    P_HOP       = [0.25, 0.5, 1.0];

    % Note: librosa.stft does not support user-supplied windows,
    %       so we do not generate tests for this case.

    counter     = 0;

    for nfft = P_NFFT
        for win_ratio = P_WIN
            for hop_ratio = P_HOP
                % Test once with no hann window (rectangular)
                hop_length  = round(hop_ratio * nfft);
                hann_w      = round(win_ratio * nfft);
            
                [F, D] = ifgram(y, nfft, hann_w, hop_length, sr);
    
                counter     = counter + 1;
                filename    = sprintf('%s/core-ifgram-%03d.mat', output_path, counter);
                display(['  `-- saving ', filename]);
                save(filename, 'wavfile', 'F', 'D', 'sr', 'nfft', 'hann_w', 'hop_length');
            end
        end
    end
end

function testISTFT(output_path)

    wavfile     = 'data/test1_22050.wav';

    [y_in, sr]     = wavread(wavfile);
    y_in           = mean(y_in, 2);        % Convert to mono

    % Test a couple of different FFT window sizes
    P_NFFT      = [128, 256, 1024];

    % And hop sizes
    P_HOP       = [64, 128, 256];

    % Note: librosa.stft does not support user-supplied windows,
    %       so we do not generate tests for this case.

    counter     = 0;

    for nfft = P_NFFT
        for hop_length = P_HOP
            % Test once with no hann window (rectangular)
            hann_w  = 0;
            D       = stft(y_in, nfft, hann_w, hop_length, sr);
            Dinv    = istft(D, nfft, hann_w, hop_length);

            counter     = counter + 1;
            filename    = sprintf('%s/core-istft-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);
            save(filename, 'D', 'Dinv', 'nfft', 'hann_w', 'hop_length');

            % And again with default hann window (nfft)
            hann_w      = nfft;
            D           = stft(y_in, nfft, hann_w, hop_length, sr);
            Dinv        = istft(D, nfft, hann_w, hop_length);

            counter     = counter + 1;
            filename    = sprintf('%s/core-istft-%03d.mat', output_path, counter);
            display(['  `-- saving ', filename]);
            save(filename, 'D', 'Dinv', 'nfft', 'hann_w', 'hop_length');
        end
    end
end

function testBeat(output_path)

    wavfile     = 'data/test2_8000.wav';

    [y, sr]     = wavread(wavfile);
    y           = mean(y, 2);               % Convert to mono

    % Generate the onset envelope first
    [t, xcr, D, onsetenv, oesr] = tempo2(y, sr);

    filename    = sprintf('%s/beat-onset-000.mat', output_path);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 'onsetenv', 'D');

    filename    = sprintf('%s/beat-tempo-000.mat', output_path);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 't', 'onsetenv');

    [beats, onsetenv_out, D, cumscore] = beat2(onsetenv, oesr);
    filename    = sprintf('%s/beat-beat-000.mat', output_path);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 'beats', 'onsetenv');

end

function testLPCBurg(output_path)
    rng(1);

    output_counter = 1;
    for m=3:20
        signal = {};
        est_coeffs = {};
        order = {};
        true_coeffs = {};

        i=1;
        for l=pow2(6:16)
            if l < m*2
                continue
            end

            % Generate random, but stable filters
            while 1
                % 0.25... Let's not play the stable filter
                % lottery forever
                coeffs = rand(1, m-1) .* 0.25;
                % Keep away from the edge
                if all(abs(roots([1 coeffs])) < 1-10*eps)
                    break
                end
            end

            % Filter some noise with them and store them for testing
            noise = randn(l,1);
            signal{i} = filter(1, [1 coeffs], noise);
            est_coeffs{i} = arburg(signal{i}, m);
            order{i} = m;
            true_coeffs{i} = [1 coeffs];

            i=i+1;
        end

        filename = sprintf('%s/core-lpcburg-%03d.mat', output_path, output_counter);
        display(['  `-- saving ', filename]);
        save(filename, 'signal', 'est_coeffs', 'order', 'true_coeffs');
        output_counter = output_counter + 1;
    end
end

function testChromafb(output_path)

    % Test three sample rates
    P_SR        = [8000, 11025, 22050];

    % Two FFT lengths
    P_NFFT      = [256, 512];

    % Two filter bank sizes
    P_NCHROMA   = [12, 24];

    % Two A440s
    P_A440      = [435.0, 440.0];

    % Two center octaves
    P_CTROCT    = [4.0, 5.0];

    % Two octave widths
    P_OCTWIDTH  = [0, 2.0];

    % Generate tests
    counter = 0;

    for sr = P_SR
        for nfft = P_NFFT
            for nchroma = P_NCHROMA
                for a440 = P_A440
                    for ctroct = P_CTROCT
                        for octwidth = P_OCTWIDTH

                            % Run the function
                            wts = fft2chromamx(nfft, nchroma, sr, a440, ctroct, octwidth);

                            % save the output
                            counter = counter + 1;

                            filename = sprintf('%s/feature-chromafb-%03d.mat', output_path, counter);
                            display(['  `-- saving ', filename]);

                            save(filename, ...
                                    'sr', 'nfft', 'nchroma', 'a440', ...
                                    'ctroct', 'octwidth', 'wts');
                        end
                    end
                end
            end
        end
    end
end
