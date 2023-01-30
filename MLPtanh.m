% Implementacao da rede MLP canonica 
% Backpropagation com gradiente descendente + termo de momento
% Funcao de ativacao tangente hiperbolica
% Usando as funcoes built-in (internas) do matlab
%
% Exemplo para disciplina de ICA
% Autor: Guilherme de A. Barreto
% Date: 25/12/2008


% X = Vetor de entrada
% d = saida desejada (escalar)
% W = Matriz de pesos Entrada -> Camada Oculta
% M = Matriz de Pesos Camada Oculta -> Camada saida
% eta = taxa de aprendizagem
% alfa = fator de momento

clear; clc;


% Carrega DADOS
%=================
 dados=load('iris_input.txt');
 alvos=load('iris_target.txt');

% dados=load('derm_input.txt'); % Vetores (padroes) de entrada
% alvos=load('derm_target.txt'); % Saidas desejadas correspondentes

% dados=load('wine_input.txt');
% alvos=load('wine_target.txt');

% dados=load('column_input.txt');
% alvos=load('column_target.txt');

 %dados=load('card_input.txt')';
 %alvos=load('card_target.txt')';

% X = xlsread('sismic', 'A1:S2584');
% 
% dados = X(:,1:end-1);
% alvos = X(:,end);
% 
% dados=dados';
% alvos=alvos';

%B = load('sensor_readings_24_r.txt');
%dados = B(:, 1:24)';
%alvos = B(:, 25:28)';

 alvos=2*alvos-1;  % Mapeia para faixa [-1,+1]

% Embaralha vetores de entrada e saidas desejadas
[LinD ColD]=size(dados);

% % Normaliza componentes para media zero e variancia unitaria
for i=1:LinD,
	mi=mean(dados(i,:));  % Media das linhas
    di=std(dados(i,:));   % desvio-padrao das linhas 
	dados(i,:)= (dados(i,:) - mi)./di;
end 
Dn=dados;

%Normaliza componentes para a faixa [-1,+1]
for i=1:LinD,
	Xmax=max(dados(i,:));  % Media das linhas
    Xmin=min(dados(i,:));   % desvio-padrao das linhas 
	dados(i,:)= 2*( (dados(i,:) - Xmin)/(Xmax-Xmin) ) - 1;
end 
Dn=dados;

% Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn=0.8;    % Porcentagem usada para treino

% DEFINE ARQUITETURA DA REDE
%=========================
Ne = 100; % No. de epocas de treinamento
Nr = 100; % No. de rodadas de treinamento/teste
Nh = 2;   % No. de neuronios na camada oculta
No = 3;   % No. de neuronios na camada de saida

 etai=0.5;   % Passo de aprendizagem inicial
 etaf=0.001;   % Passo de aprendizagem final
%eta=0.1;

%% Inicio do Treino
tic
for r=1:Nr,  % LOOP DE RODADAS TREINO/TESTE

    Rodada=r,

    I=randperm(ColD);
    Dn=Dn(:,I);
    alvos=alvos(:,I);   % Embaralha saidas desejadas tambem p/ manter correspondencia com vetor de entrada

    J=floor(ptrn*ColD);

    % Vetores para treinamento e saidas desejadas correspondentes
    P = Dn(:,1:J); T1 = alvos(:,1:J);
    [lP cP]=size(P);   % Tamanho da matriz de vetores de treinamento

    % Vetores para teste e saidas desejadas correspondentes
    Q = Dn(:,J+1:end); T2 = alvos(:,J+1:end);
    [lQ cQ]=size(Q);   % Tamanho da matriz de vetores de teste

    % Inicia matrizes de pesos
    WW=0.5*(2*rand(Nh,lP+1)-1);   % Pesos entrada -> camada oculta
    MM=0.5*(2*rand(No,Nh+1)-1);   % Pesos camada oculta -> camada de saida
%     WW=[0.771168581569,0.030350433211,0.0997309773693,0.178536646617,0.665419690155,0.708732437672,0.666079960608,0.805897937997,0.726643918886,0.704344719511,0.921700816099,0.0256161824919,0.531179141966,0.52783902666,0.390521068773,0.487602875795,0.141533487077,0.753317297321,0.00381608121275,0.136876942654,0.490775179346,0.458439275836,0.988908973052,0.593110079222,0.401101483219,0.312628459797,0.346523801492,0.0255316733502,0.110833996027,0.786971228564,0.625438469753,0.744361144371,0.477753451782,0.602264102829,0.252776243842; 0.410330246394,0.420451139296,0.522298142557,0.264881958377,0.871074442692,0.148158318898,0.0968657150384,0.0220726509681,0.975044821377,0.578312889942,0.704741257105,0.586308169917,0.0814118008508,0.288136900071,0.716879495288,0.593677300771,0.934394051756,0.360827862453,0.433884249271,0.292577495003,0.349958521943,0.752878289089,0.625404719089,0.177113726818,0.750406623236,0.0841167290155,0.749864564161,0.973729849781,0.477585265635,0.775559532817,0.829068059488,0.146875810878,0.541753423652,0.249791327515,0.242841537689; 0.437723943236,0.826313970064,0.858894864963,0.44599543067,0.845203266407,0.331298502316,0.13392843126,0.935144192975,0.96845132437,0.761408686061,0.99578662915,0.185876117174,0.0199013510812,0.482007621081,0.10208750195,0.784645268128,0.533021420954,0.491021978432,0.606391504224,0.622011486731,0.147057480247,0.595070515571,0.350155197247,0.0584001303922,0.530991502353,0.374180054466,0.844175415507,0.056208420571,0.694924537416,0.596700351497,0.742807618223,0.367639466826,0.91651893962,0.93381819545,0.682410928738; 0.280479307883,0.0157275847233,0.333516443769,0.41087042932,0.499305576784,0.828829012266,0.129209151552,0.618210129262,0.257642507207,0.197618634998,0.376398406167,0.128012444884,0.50516115851,0.243591079136,0.0352670415469,0.733167278456,0.342449011906,0.540543108499,0.908024541525,0.168469413728,0.465436522134,0.591627511471,0.483585298752,0.618116128546,0.677772476653,0.322015114744,0.108033505784,0.719131717793,0.44678095097,0.047442959178,0.373814904305,0.707096655251,0.17348480419,0.759104013331,0.261152051045; 0.18252191608,0.645843549467,0.692535885467,0.450627035671,0.688588525955,0.107355724139,0.327655601468,0.907693868926,0.61085503344,0.640547025781,0.673862307646,0.603804610485,0.144088423412,0.694132291104,0.281416581143,0.768479276341,0.831197457775,0.935672826104,0.853188325117,0.536180240817,0.581307418915,0.0337897064322,0.903596005358,0.738062045415,0.608797289249,0.0560404067189,0.87111572403,0.841973780115,0.053322394869,0.189490562859,0.76788996708,0.926676716156,0.655568428643,0.138580200327,0.117426894194; 0.593810721577,0.176797545597,0.436348854302,0.715194247996,0.26972606977,0.286054619256,0.719985827208,0.801797877439,0.816926120695,0.0773105235199,0.357968799005,0.381604881204,0.633238388055,0.837588034495,0.342095750543,0.603279381806,0.316570008321,0.592129851036,0.926406359266,0.111680179421,0.00877552386782,0.490229646438,0.289667681926,0.444730126506,0.579236187311,0.22260012814,0.240353643075,0.623679158103,0.17561022992,0.481134272405,0.423716303158,0.399907183554,0.240033991747,0.251299285447,0.587090506026; 0.230134780626,0.875257979089,0.460854549641,0.582415819439,0.662677316304,0.617655115955,0.929533861545,0.675610981731,0.993769945574,0.2914752603,0.824699867435,0.730671982155,0.404004087394,0.096696832728,0.183667658914,0.90234336299,0.684901772852,0.144096315905,0.826781423682,0.715387819668,0.523085159959,0.492283428317,0.80757972496,0.99243739992,0.895380460143,0.659393620053,0.428572231172,0.013489306445,0.714773421043,0.196887469011,0.0876916652022,0.833817053975,0.963226149773,0.941899232539,0.500401288038; 0.244448059818,0.438541360869,0.564652130736,0.108361285696,0.228128692241,0.158930501975,0.144946687457,0.118976084105,0.63104555599,0.982659519642,0.558546622544,0.493085094492,0.281183119063,0.844682089912,0.571885152521,0.673758413025,0.857647712276,0.485100225306,0.0794867128504,0.933182876526,0.00460577244154,0.409217424881,0.717259979675,0.988478400739,0.356481222602,0.379908268983,0.118276788908,0.877991179879,0.397760223317,0.156073286271,0.123722352611,0.40158032924,0.360593538434,0.495600453343,0.556819336748; 0.462592725857,0.795943485944,0.422168253652,0.381839127458,0.570215180316,0.606535572841,0.0433727400579,0.965642153735,0.547677825926,0.821220333605,0.250146894367,0.218853618586,0.272767573256,0.404603720365,0.174728180829,0.65653519037,0.386944551667,0.377079862346,0.581246441035,0.00893447827964,0.161776445881,0.976725928009,0.832672045488,0.719068520572,0.384625246927,0.396525109371,0.397513205371,0.00444266433103,0.667859411644,0.713131496549,0.601062502061,0.0574721321731,0.934125433645,0.846163276046,0.466180499395; 0.0956533235012,0.645408084451,0.373675367038,0.361893801187,0.349116551852,0.601886982844,0.914520661772,0.348762409924,0.649823591881,0.585108749841,0.922758576424,0.80339396177,0.642315471378,0.396127455121,0.714138218069,0.521031078659,0.969339029849,0.68107467549,0.822070955682,0.546552141452,0.901841375931,0.248005273867,0.224637889408,0.489007278108,0.745323154025,0.64624970483,0.518789069969,0.287898962986,0.717870900276,0.256220939688,0.305333344408,0.737519465265,0.489652702813,0.59297617599,0.150589870825];
%     MM=[0.963958951628,0.258100008712,0.88684642589,0.227879939707,0.978146654544,0.710822913195,0.800702074915,0.399773104768,0.986571838142,0.312883647304,0.635460245719; 0.180349798026,0.139055418381,0.104416729931,0.93197995002,0.787019987491,0.444929768539,0.934619827166,0.155435171516,0.39892767528,0.777438434669,0.407771474406; 0.415170341923,0.767936707366,0.712240704201,0.629515510811,0.267190204592,0.665768578493,0.57249873065,0.986166041338,0.492656767132,0.0822851886425,0.96716551388; 0.150791783887,0.357511794361,0.700727826311,0.132576803273,0.218332606935,0.516124758178,0.508810704811,0.581515757172,0.535330792207,0.304624622364,0.826028069866; 0.0537702362303,0.716360323465,0.867956470637,0.744401989851,0.164243431373,0.439351080656,0.173612581181,0.906651911748,0.0986807477189,0.527326911002,0.783393212028; 0.489714547288,0.632396277335,0.684233164733,0.906799673991,0.582120762012,0.703647136085,0.197416173386,0.973626100446,0.733870194635,0.156361235378,0.963283006085];
    %%% ETAPA DE TREINAMENTO
    Tmax=Ne*cP;  % No. max. iteracoes de treinamento
    T=0;
    for t=1:Ne,
        Epoca=t;
        I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento
        EQ=0;
        for tt=1:cP,   % Inicia LOOP de epocas de treinamento
            % CAMADA OCULTA
            X  = [-1; P(:,tt)];   % Constroi vetor de entrada com adicao da entrada x0=-1
            %X=[-1.0;0.33333;0.33333;-0.33333;-1.0;-0.33333;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;0.33333;0.33333;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-1.0;-0.33333;-1.0;1.0;-1.0;-1.0;-1.0;0.33333;-1.0;0.65333];
            %X=[-1.0;1.0;1.0;1.0;1.0;1.0;-1.0;-1.0;-1.0;1.0;1.0;1.0;-1.0;-1.0;-0.33333;-1.0;-1.0;0.33333;-0.33333;0.33333;0.33333;0.33333;0.33333;0.33333;0.33333;-1.0;-0.33333;-1.0;-1.0;-1.0;-1.0;-1.0;0.33333;-1.0;0.12];
            %X = [-1.0;0.33333;-0.33333;-0.33333;-1.0;-1.0;-1.0;-1.0;-1.0;-0.33333;-0.33333;1.0;-1.0;-1.0;-0.33333;-1.0;-1.0; 0.33333;0.33333;0.33333;0.33333;0.33333;0.33333;-1.0;-0.33333;-1.0;-0.33333;-1.0;-1.0;-1.0;-1.0;-1.0;-0.33333;-1.0;0.066667];
            Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
            Yi = (1-exp(-Ui))./(1+exp(-Ui)); % Saida entre [-1,1]

            % CAMADA DE SAIDA
            Y  = [-1; Yi];        % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
            Uk = MM * Y;          % Ativacao (net) dos neuronios da camada de saida
            Ok = (1-exp(-Uk))./(1+exp(-Uk)); % Saida entre [-1,1]

            % CALCULO DO ERRO         1`[~2.40530117101E~06,~0.000105937694683,~8.00974591958E~06,0.0311246109181,~6.92718397251E~05,~3.16634847392E~06]
            Ek = T1(:,tt) - Ok;           % erro entre a saida desejada e a saida da rede 
            EQ = EQ + 0.5*sum(Ek.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

            %%% CALCULO DOS GRADIENTES LOCAIS
            Dk = 0.5*(1 - Ok.^2);  % derivada da sigmoide logistica (camada de saida)
            DDk = Ek.*Dk;       % gradiente local (camada de saida)

            Di = 0.5*(1 - Yi.^2); % derivada da sigmoide logistica (camada oculta)
            DDi = Di.*(MM(:,2:end)'*DDk);    % gradiente local (camada oculta)

            T=(t-1)*cP+tt;   % Iteracao atual
            eta=etai-((etai-etaf)/Tmax)*T;  % Passo de aprendizagem na iteracao T
            
            MM = MM + eta*DDk*Y';      % AJUSTE DOS PESOS - CAMADA DE SAIDA
            WW = WW + eta*DDi*X';    % AJUSTE DOS PESOS - CAMADA OCULTA
        end   % Fim de uma epoca

        EQM(t)=EQ/cP;  % MEDIA DO ERRO QUADRATICO POR EPOCA
    end   % Fim do loop de treinamento


    %% ETAPA DE GENERALIZACAO  %%%
    EQ2=0; HID2=[]; OUT2=[];
    for tt=1:cQ,
        % CAMADA OCULTA
        X = [-1; Q(:,tt)];      % Constroi vetor de entrada (x0=-1)
        Ui = WW * X;            % Ativacao dos neuronios da camada oculta
        Yi = (1-exp(-Ui))./(1+exp(-Ui));

        % CAMADA DE SAIDA
        Y=[-1; Yi];           % Constroi vetor de entrada (y0=-1)
        Uk = MM * Y;          % Ativacao dos neuronios da camada de saida
        Ok = (1-exp(-Uk))./(1+exp(-Uk));
        OUT2=[OUT2 Ok];       % Armazena saida da rede

        % CALCULO DO ERRO DE GENERALIZACAO
        Ek = T2(:,tt) - Ok;
        EQ2 = EQ2 + 0.5*sum(Ek.^2);
    end

    % MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TESTE)
    EQM2=EQ2/cQ;

    % CALCULA TAXA DE ACERTO GLOBAL E MATRIZ DE CONFUSAO
    count_OK=0;  % Zera contador de acertos
    CC=zeros(No); % Inicia matriz de confusao
    for t=1:cQ,
        [T2max Ireal]=max(T2(:,t));  % Indice da saida desejada de maior valor
        [OUT2_max Ipred]=max(OUT2(:,t)); % Indice do neuronio cuja saida eh a maior
        if Ireal==Ipred,   % Acerto se os dois indices coincidem
            count_OK=count_OK+1;
        end
        CC(Ireal,Ipred)=CC(Ireal,Ipred)+1;
    end

    Tx_OK(r)=100*(count_OK/cQ);  % Taxa de acerto global por realizacao

end % FIM DO LOOP DE RODADAS TREINO/TESTE

Tsim=toc

Tx_media=mean(Tx_OK),  % Taxa media de acerto global
Tx_std=std(Tx_OK), % Desvio padrao da taxa media de acerto 
Tx_mi=min(Tx_OK),
Tx_ma=max(Tx_OK),
Tx_mediana=median(Tx_OK)

% CPN

%A = load('dermatology_.txt');
%A = load('dados_iris3.txt');
%A = load('Wine_.txt');
%A = load('sism_.txt');
%A = load('sens2_.txt');
% plot (EQM)
%  hold on
%  plot (A)
%  title('Curvas de aprendizado dos modelos HTCPN-MLP e a MLP/Matlab.')
%  %title('Wall-Following Robot Navigation dataset learning curve')
%  xlabel('Epoch')
%  ylabel ('Mean square error')
%  legend('MLP-MAT', 'HTCPN-MLP')

% Plota Curva de Aprendizagem
%plot(EQM)
% figure
% set(gcf,'Name','Mean square error (MSE) - dataset Dermatology ')
% subplot(2,1,1);
% plot(EQM)
% title('Subplot 1 : RNA mean square error - dataset Dermatology')
% 
% subplot(2,1,2);
% plot(A(1:100))
% title('Subplot 2 : HCTPN-MLP mean square error - dataset Dermatology')

%  plot (EQM)
%  hold on
%  plot (A)
%  title('Curvas de aprendizado dos modelos HTCPN-MLP e a MLP/Matlab.')
%  %title('The learning curve of the Seismic-bumps dataset.')
%  xlabel('Epoch')
%  ylabel ('Mean square error')
%  legend('MLP-MAT', 'HTCPN-MLP')