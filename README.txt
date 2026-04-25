Alterations based on paper

% Observer

ObservationInfo = rlNumericSpec([100 1]);
ObservationInfo.Name = 'Parcel States';
ObservationInfo.Description = 'x_box, y_box, w_box, h_box';

** Location of the GC of the parcel: x and y
** Width and height: w and h







% Initial observer

InitialObservation = zeros(100,1);

            for ii = 1:this.n_boxes_tot
                if this.index_AMS(ii) > 0
                    k = this.index_AMS(ii);
                    idx = (k-1)*4 + 1;
            
                    InitialObservation(idx)   = this.x_boxes(ii);
                    InitialObservation(idx+1) = this.y_boxes(ii);
                    InitialObservation(idx+2) = this.d_boxes(ii);
                    InitialObservation(idx+3) = this.d_boxes(ii);
                end
            end  

** If a box is on an AMS: Set the bundle of 4 var of vector to the characteristics of the box







% Reward

	packages_in_reward_area = [];
        
            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) > 0 && this.y_boxes(ii) <= this.d_AMS*5
                    packages_in_reward_area = [packages_in_reward_area; ii];
                end
            end
        
            K = length(packages_in_reward_area);
        
            if K < 2
                Reward = 0;
                return;
            end
        
            % Sort packages by y-position
            y_values = this.y_boxes(packages_in_reward_area);
            [~, sort_idx] = sort(y_values, 'ascend');
            packages_sorted = packages_in_reward_area(sort_idx);
        
            scores = zeros(K-1,1);
            all_gaps_positive = true;
        
            for kk = 1:K-1
        
                current_box = packages_sorted(kk);
                next_box    = packages_sorted(kk+1);
        
                y_max_current = this.y_boxes(current_box) + this.d_boxes(current_box)/2;
                y_min_next    = this.y_boxes(next_box)    - this.d_boxes(next_box)/2;
        
                d_k = y_min_next - y_max_current;
        
                % Local score for this neighboring package pair
                local_score = 10 * K * atan(d_k) + K;
        
                % Clip maximum score
                local_score = min(local_score, 6);
        
                scores(kk) = local_score;
        
                if d_k <= 0
                    all_gaps_positive = false;
                end
        
            end
        
            % Worst-case reward
            Reward = min(scores);
        
            % Bonus if all package gaps are positive and at least 3 packages
            % are currently evaluated
            if all_gaps_positive && K >= 3
                Reward = Reward + 20;
            end
                     
        end

** For each pair of consecutive parcels, the signed gap in the y-direction is computed as:

** d_k = y_min(p_{k+1}) - y_max(p_k)

** This means:

** d_k > 0  -> there is a real gap between the parcels
** d_k < 0  -> the parcels overlap in the y-direction


** Each gap is converted into a local score:
** ls_k = min(10 * K * atan(d_k) + K, 6)

** where K is the number of parcels in the reward area. The atan function makes the score saturate, so very large gaps do not produce unbounded rewards.
** The immediate reward is then the worst local score:

** r_t = min(ls_k)

** So the agent is not rewarded for the average spacing, but for the worst parcel pair. This forces all parcels to be separated properly.

** Additionally, a bonus of +20 is given if all gaps are positive and at least three parcels are currently evaluated.