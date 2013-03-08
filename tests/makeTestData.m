function makeTestData(source_path, output_path)
% makeTestData(source_path, audio_file, output_path)
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
%       dctfb
%       melfb
%       localmax
%       stft
%       istft
%

    % Make sure we have the path to DPWE code
    addpath(source_path);

    %% hz_to_mel tests
    display('Generating hz_to_mel tests');
    makeTestHz2Mel(output_path);

    display('Generating mel_to_hz tests');
    makeTestMel2Hz(output_path);

    %% Done!
    display('Done.');
end

function makeTestHz2Mel(output_path)

    % Parameters to sweep
    P_FREQUENCIES = {[440], [2.^(1:13)]};
    P_HTK         = {0, 1};

    counter     = 0;
    for i = 1:length(P_FREQUENCIES)
        f = P_FREQUENCIES{i};

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

function makeTestMel2Hz(output_path)

    % Parameters to sweep
    P_MELS      = {[5], [2.^(-2:9)]};
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
