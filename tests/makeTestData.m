function testData(source_path, output_path)
% testData(source_path, audio_file, output_path)
%   source_path = path to DPWE code
%   output_ptah = directory to store generated files
%
% CREATED:2013-03-08 14:32:21 by Brian McFee <brm2132@columbia.edu>
%   Generate the test suite data for librosa comparison.
%
%   Validated methods:
%       hz_to_mel
%       mel_to_hz
%       hz_to_octs
%
%       dctfb               
%       melfb
%
%       localmax
%
%       stft
%       istft
%
%       load
%       resample
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

%     display('melfb');
%     testMelfb(output_path);

    display('resample');
    testResample(output_path);

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

            filename = sprintf('%s/hz_to_mel-%03d.mat', output_path, counter);
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

            filename = sprintf('%s/mel_to_hz-%03d.mat', output_path, counter);
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

        filename = sprintf('%s/hz_to_octs-%03d.mat', output_path, counter);
        display(['  `-- saving ', filename]);

        save(filename, 'f', 'result');
    end
end

function testLoad(output_path)

    % Test: load a wav file
    %       get audio stream (floats) and sample rate
    %       preserve stereo or convert to mono
    wavfile          = 'data/test1_44100.wav';
    [y, sr]         = wavread(wavfile);
    mono            = 0;

    % Stereo output
    counter = 1;

    filename = sprintf('%s/load-%03d.mat', output_path, counter);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 'mono', 'y', 'sr');

    % Mono output
    counter = 2;
    mono    = 1;
    y       = mean(y, 2);
    filename = sprintf('%s/load-%03d.mat', output_path, counter);
    display(['  `-- saving ', filename]);
    save(filename, 'wavfile', 'mono', 'y', 'sr');

end

function testMelfb(output_path)

    % Test three sample rates
    P_SR        = [8000, 11025, 22050];

    % Two FFT lengths
    P_NFFT      = [256, 516];

    % Three filter bank sizes
    P_NFILTS    = [20, 40, 120];

    % Two widths
    P_WIDTH     = [1.0, 3.0];

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
            [wts, frqs] = fft2melmx(sr, nfft, nfilts, width, fmin, fmax, htk, 0);

            % save the output
            counter = counter + 1;

            filename = sprintf('%s/melfb-%03d.mat', output_path, counter);
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

    P_SR            = [8000, 22050, 44100];

    counter = 0;

    for sr_out = P_SR

        y_out       = resample(y_in, sr_out, sr_in);

        counter     = counter + 1;
        filename    = sprintf('%s/resample-%03d.mat', output_path, counter);

        display(['  `-- saving ', filename]);

        save(filename, 'wavfile', 'y_in', 'sr_in', 'y_out', 'sr_out');
    end
end

function testSTFT(output_path)

    wavfile     = 'data/test1_22050.wav';

    [y, sr]     = wavread(wavfile);
    y           = mean(y_in, 2);        % Convert to mono





end
