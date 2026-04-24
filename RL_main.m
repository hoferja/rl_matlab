
clear all
close all
clc

%%

% load the RL environment

env=RL_environment();

doTrain = false;
nMaxEpisodes = 25000;

if doTrain==true

    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);

    numObs = prod(obsInfo.Dimension);
    criticLayerSizes = [512 256 128];
    actorLayerSizes = [512 256 128];

    % Critic

    criticNetwork = [
        featureInputLayer(numObs)
        fullyConnectedLayer(criticLayerSizes(1), ...
        Weights=sqrt(2/numObs)*...
        (rand(criticLayerSizes(1),numObs)-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(1),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(2), ...
        Weights=sqrt(2/criticLayerSizes(1))*...
        (rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(2),1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(3), ...
        Weights=sqrt(2/criticLayerSizes(2))*...
        (rand(criticLayerSizes(3),criticLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(3),1))
        reluLayer
        fullyConnectedLayer(1, ...
        Weights=sqrt(2/criticLayerSizes(3))* ...
        (rand(1,criticLayerSizes(3))-0.5), ...
        Bias=1e-3)
        ];

    criticNetwork = dlnetwork(criticNetwork);
    summary(criticNetwork)

    critic = rlValueFunction(criticNetwork,obsInfo);

    % Actor

    inPath = [
        featureInputLayer(numObs,Name="netOin")
        fullyConnectedLayer(actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        reluLayer(Name="relulast")
        ];

    meanPath = [
        fullyConnectedLayer(actorLayerSizes(3),Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension),Name="meanOutLyr")
        tanhLayer(Name="thmeanOutLyr");
        ];

    sdevPath = [
        fullyConnectedLayer(actorLayerSizes(3),Name="StdLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension))
        reluLayer
        softplusLayer(Name="stdOutLyr")
        ];

    % Add layers to network object
    net = layerGraph(inPath);
    net = addLayers(net,meanPath);
    net = addLayers(net,sdevPath);

    % Connect layers
    net = connectLayers(net,"relulast","MeanLyr/in");
    net = connectLayers(net,"relulast","StdLyr/in");

    net = dlnetwork(net);
    summary(net)

    actor = rlContinuousGaussianActor(net, obsInfo, actInfo, ...
        ActionMeanOutputNames="thmeanOutLyr",...
        ActionStandardDeviationOutputNames="stdOutLyr",...
        ObservationInputNames="netOin");

    % Train

    actorOpts = rlOptimizerOptions(LearnRate=1e-4);
    criticOpts = rlOptimizerOptions(LearnRate=1e-4);

    agentOpts = rlPPOAgentOptions(...
        ExperienceHorizon=500,...
        ClipFactor=0.02,...
        EntropyLossWeight=0.01,...
        ActorOptimizerOptions=actorOpts,...
        CriticOptimizerOptions=criticOpts,...
        NumEpoch=3,...
        AdvantageEstimateMethod="gae",...
        GAEFactor=0.95,...
        SampleTime=0.01,...
        DiscountFactor=0.99);

    agent = rlPPOAgent(actor,critic,agentOpts);

    trainOpts = rlTrainingOptions(...
        MaxEpisodes=nMaxEpisodes,...
        MaxStepsPerEpisode=1000,...
        Plots="training-progress",...
        StopTrainingCriteria="AverageReward",...
        StopTrainingValue=40000,...
        ScoreAveragingWindowLength=100);

    trainingStats = train(agent, env, trainOpts);

    save("agent_trained","agent");
    save("trainingStats","trainingStats");

else

    % load saved agent
    load("agent_trained.mat","agent");
    load("trainingStats.mat","trainingStats");

end

%% Print analytics (self-made)
%% Plot analytics
figure;

% Plot episode reward over time
subplot(2, 2, 1);
plot(trainingStats.EpisodeReward);
xlabel('Episode');
ylabel('Reward');
title('Episode Reward Over Time');

% Plot episode length over time
subplot(2, 2, 2);
plot(trainingStats.EpisodeSteps);
xlabel('Episode');
ylabel('Steps');
title('Episode Length Over Time');

% Plot smoothed episode reward over time
subplot(2, 2, 3);
plot(trainingStats.SmoothedEpisodeReward);
xlabel('Episode');
ylabel('Smoothed Reward');
title('Smoothed Episode Reward Over Time');

% Plot policy loss and value loss over time
subplot(2, 2, 4);
yyaxis left;
plot(trainingStats.PolicyLoss);
ylabel('Policy Loss');
yyaxis right;
plot(trainingStats.ValueLoss);
ylabel('Value Loss');
xlabel('Training Iteration');
title('Policy and Value Losses');

% Adjust the layout and display the legend
sgtitle('Training Analytics');
tight_layout(pad=2);

%% simulation of the sorting after training

env.reset();

plot(env)

rng(10)
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 10;
experience = sim(env, agent, simOptions);
